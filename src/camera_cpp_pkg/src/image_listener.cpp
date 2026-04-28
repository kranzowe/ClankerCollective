#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

using std::placeholders::_1;

// ---- BGR -> HLS_FULL  (H, L, S each in [0, 255]) — integer math, no fmodf ----
static inline void bgr2hls_full_int(int r, int g, int b,
                                    int &H, int &L, int &S)
{
    int cmax = std::max({r, g, b});
    int cmin = std::min({r, g, b});
    int diff = cmax - cmin;
    int sum  = cmax + cmin;

    L = sum >> 1;

    if (diff == 0) {
        H = 0;
        S = 0;
        return;
    }

    // Saturation, integer: matches OpenCV HLS_FULL scaling.
    if (sum <= 255)
        S = (255 * diff) / sum;
    else
        S = (255 * diff) / (510 - sum);

    // Hue in degrees [0, 360), then scaled to [0, 255].
    int h_deg;
    if (cmax == r)
        h_deg = (60 * (g - b)) / diff + (g < b ? 360 : 0);
    else if (cmax == g)
        h_deg = (60 * (b - r)) / diff + 120;
    else
        h_deg = (60 * (r - g)) / diff + 240;

    H = (h_deg * 255 + 180) / 360;   // round-half-up scale 360 -> 255
    if (H < 0)   H = 0;
    if (H > 255) H = 255;
}

// ---- 5x5 Gaussian blur [1,4,6,4,1]/16 separable, scratch buffers passed in ----
static void gaussian_blur_bgr(const uint8_t *src, std::vector<uint8_t> &dst,
                              std::vector<int> &tmp, int w, int h)
{
    static const int K[5] = {1, 4, 6, 4, 1};
    const int N = w * h * 3;
    if ((int)dst.size() != N) dst.resize(N);
    if ((int)tmp.size() != w * h) tmp.resize(w * h);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sum = 0;
                for (int k = -2; k <= 2; k++)
                    sum += src[(y * w + std::clamp(x + k, 0, w - 1)) * 3 + c] * K[k + 2];
                tmp[y * w + x] = sum;
            }
        }
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sum = 0;
                for (int k = -2; k <= 2; k++)
                    sum += tmp[std::clamp(y + k, 0, h - 1) * w + x] * K[k + 2];
                dst[(y * w + x) * 3 + c] = (uint8_t)std::clamp(sum / 256, 0, 255);
            }
        }
    }
}

// ---- Integral image build (scratch passed in) ----
static void build_ii(const std::vector<uint8_t> &mask, std::vector<int> &ii,
                     int w, int h)
{
    const int W1 = w + 1, H1 = h + 1;
    if ((int)ii.size() != W1 * H1) ii.assign(W1 * H1, 0);
    // Top row + left column of zero are always 0; we still need to overwrite the data area.
    for (int x = 0; x <= w; x++) ii[x] = 0;
    for (int y = 0; y < h; y++) {
        const int row_off  = (y + 1) * W1;
        const int prev_off = y * W1;
        ii[row_off] = 0;
        for (int x = 0; x < w; x++) {
            ii[row_off + x + 1] = mask[y * w + x]
                + ii[prev_off + x + 1] + ii[row_off + x] - ii[prev_off + x];
        }
    }
}

static inline int ii_sum(const std::vector<int> &ii, int W1,
                         int y1, int x1, int y2, int x2)
{
    return ii[y2 * W1 + x2] - ii[y1 * W1 + x2] - ii[y2 * W1 + x1] + ii[y1 * W1 + x1];
}

static void erode5(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst,
                   std::vector<int> &ii, int w, int h)
{
    build_ii(src, ii, w, h);
    const int W1 = w + 1;
    if ((int)dst.size() != w * h) dst.resize(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int y1 = std::max(0, y - 2), y2 = std::min(h, y + 3);
            int x1 = std::max(0, x - 2), x2 = std::min(w, x + 3);
            dst[y * w + x] = (ii_sum(ii, W1, y1, x1, y2, x2) == (y2 - y1) * (x2 - x1)) ? 1 : 0;
        }
    }
}

static void dilate5(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst,
                    std::vector<int> &ii, int w, int h)
{
    build_ii(src, ii, w, h);
    const int W1 = w + 1;
    if ((int)dst.size() != w * h) dst.resize(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int y1 = std::max(0, y - 2), y2 = std::min(h, y + 3);
            int x1 = std::max(0, x - 2), x2 = std::min(w, x + 3);
            dst[y * w + x] = (ii_sum(ii, W1, y1, x1, y2, x2) > 0) ? 1 : 0;
        }
    }
}

// ================================================================
class ImageListener : public rclcpp::Node
{
public:
    ImageListener() : Node("image_listener")
    {
        // SensorDataQoS: keep_last(1) + best_effort — always process the latest frame.
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&ImageListener::callback, this, _1));

        path_pub_ = create_publisher<nav_2d_msgs::msg::Path2D>("/line_path", 10);

        declare_parameter("fh", 120);
        declare_parameter("fs", 180);
        declare_parameter("fl", 120);   // lightness center
        declare_parameter("th", 45);
        declare_parameter("ts", 80);
        declare_parameter("tl", 100);   // lightness tolerance

        recompute_hls_bounds();
        // Refresh cached HLS bounds once a second (same cadence as Python).
        param_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ImageListener::recompute_hls_bounds, this));
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<nav_2d_msgs::msg::Path2D>::SharedPtr   path_pub_;
    rclcpp::TimerBase::SharedPtr                             param_timer_;

    int prev_hband_x_ = -1;

    // cached HLS thresholds (refreshed by param_timer_)
    int lo_h_ = 0, hi_h_ = 255;
    int lo_l_ = 0, hi_l_ = 255;
    int lo_s_ = 0, hi_s_ = 255;

    // hoisted scratch buffers (reused across frames)
    std::vector<uint8_t> blur_buf_;
    std::vector<int>     blur_tmp_;
    std::vector<uint8_t> mask_a_, mask_b_, mask_;
    std::vector<int>     ii_buf_;

    void recompute_hls_bounds()
    {
        int fh = get_parameter("fh").as_int();
        int fs = get_parameter("fs").as_int();
        int fl = get_parameter("fl").as_int();
        int th = get_parameter("th").as_int();
        int ts = get_parameter("ts").as_int();
        int tl = get_parameter("tl").as_int();
        lo_h_ = fh - th; hi_h_ = fh + th;
        lo_l_ = fl - tl; hi_l_ = fl + tl;
        lo_s_ = fs - ts; hi_s_ = fs + ts;
    }

    std::pair<float, float> image_to_robot(int px, int py, int width, int height)
    {
        float x_norm = (px - width  / 2.0f) / (width  / 2.0f);
        float y_norm = std::pow((height - py) / (float)height, 1.5f);
        return {y_norm * 1.0f, x_norm * 1.4f};
    }

    // ---- BFS blobs inside a rectangle of the mask ----
    struct Blob { int area, min_x, max_x, min_y, max_y; };

    std::vector<Blob> find_blobs(const std::vector<uint8_t> &mask, int full_w,
                                 int xs, int xe, int ys, int ye)
    {
        std::vector<Blob> blobs;
        int rw = xe - xs, rh = ye - ys;
        if (rw <= 0 || rh <= 0) return blobs;

        std::vector<uint8_t> vis(rw * rh, 0);
        std::vector<std::pair<int, int>> q;
        q.reserve(256);

        for (int y = ys; y < ye; y++) {
            for (int x = xs; x < xe; x++) {
                int lx = x - xs, ly = y - ys;
                if (!mask[y * full_w + x] || vis[ly * rw + lx]) continue;

                q.clear();
                q.push_back({x, y});
                vis[ly * rw + lx] = 1;
                Blob blob{0, x, x, y, y};

                for (size_t qi = 0; qi < q.size(); qi++) {
                    auto [cx, cy] = q[qi];
                    blob.area++;
                    blob.min_x = std::min(blob.min_x, cx);
                    blob.max_x = std::max(blob.max_x, cx);
                    blob.min_y = std::min(blob.min_y, cy);
                    blob.max_y = std::max(blob.max_y, cy);

                    const int dx[4] = {1, -1, 0, 0};
                    const int dy[4] = {0, 0, 1, -1};
                    for (int k = 0; k < 4; k++) {
                        int nx = cx + dx[k], ny = cy + dy[k];
                        if (nx < xs || nx >= xe || ny < ys || ny >= ye) continue;
                        int nlx = nx - xs, nly = ny - ys;
                        if (mask[ny * full_w + nx] && !vis[nly * rw + nlx]) {
                            vis[nly * rw + nlx] = 1;
                            q.push_back({nx, ny});
                        }
                    }
                }
                blobs.push_back(blob);
            }
        }
        return blobs;
    }

    // ---- Main Callback ----
    void callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        const auto t0 = std::chrono::steady_clock::now();

        const bool is_rgb8 = (msg->encoding == "rgb8");
        if (msg->encoding != "bgr8" && !is_rgb8) {
            RCLCPP_WARN(get_logger(), "Expected bgr8 or rgb8, got %s", msg->encoding.c_str());
            return;
        }

        const int orig_height = (int)msg->height;
        const int width       = (int)msg->width;

        // Crop top 25% (matching Python crop_ratio = 0.75).
        const int start_y = (int)(orig_height * 0.25);
        const int height  = orig_height - start_y;
        const uint8_t *crop_ptr = msg->data.data() + start_y * width * 3;

        // Gaussian blur (channel-agnostic — encoding is preserved through it).
        gaussian_blur_bgr(crop_ptr, blur_buf_, blur_tmp_, width, height);

        // ---- Build HLS_FULL mask. Pick R/G/B based on encoding without copying. ----
        const int N = width * height;
        if ((int)mask_a_.size() != N) mask_a_.resize(N);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const uint8_t *p = &blur_buf_[(y * width + x) * 3];
                int r, g, b;
                if (is_rgb8) { r = p[0]; g = p[1]; b = p[2]; }
                else         { b = p[0]; g = p[1]; r = p[2]; }

                int H, L, S;
                bgr2hls_full_int(r, g, b, H, L, S);
                bool in_range = H >= lo_h_ && H <= hi_h_ &&
                                L >= lo_l_ && L <= hi_l_ &&
                                S >= lo_s_ && S <= hi_s_;
                mask_a_[y * width + x] = in_range ? 1 : 0;
            }
        }

        // ---- Morphological OPEN (erode -> dilate) then CLOSE (dilate -> erode) ----
        erode5 (mask_a_, mask_b_, ii_buf_, width, height);
        dilate5(mask_b_, mask_a_, ii_buf_, width, height);   // OPEN  -> mask_a_
        dilate5(mask_a_, mask_b_, ii_buf_, width, height);
        erode5 (mask_b_, mask_,   ii_buf_, width, height);   // CLOSE -> mask_

        // ---- Horizontal bands (8 bands, orig_height for band_h, matches Python) ----
        nav_2d_msgs::msg::Path2D path;
        const int num_bands = 8;
        const int band_h = orig_height / num_bands;
        int prev_x = prev_hband_x_;
        std::vector<std::pair<int, int>> chosen_pts;
        chosen_pts.reserve(num_bands);

        for (int i = 0; i < num_bands; i++) {
            int y1 = i * band_h;
            int y2 = (i == num_bands - 1) ? orig_height : (i + 1) * band_h;
            if (y1 >= height) break;
            y2 = std::min(y2, height);

            auto blobs = find_blobs(mask_, width, 0, width, y1, y2);

            float best_score = 0.0f;
            int best_cx = -1, best_cy = -1;

            for (auto &b : blobs) {
                if (b.area < 45) continue;
                int cx = (b.min_x + b.max_x) / 2;
                int cy = (b.min_y + b.max_y) / 2;

                float x_norm = (cx - width / 2.0f) / (width / 2.0f);
                float x_bias = (x_norm + 1.0f) / 2.0f;
                float cont_p = (prev_x != -1) ? std::abs(cx - prev_x) * 2.0f : 0.0f;
                float score  = b.area * 1.0f + x_bias * 2000.0f - cont_p;
                if (x_bias > 0.7f) score += 1500.0f;

                if (score > best_score) {
                    best_score = score;
                    best_cx = cx;
                    best_cy = cy;
                }
            }

            if (best_cx != -1) {
                chosen_pts.push_back({best_cx, best_cy});
                prev_x = best_cx;
            } else {
                chosen_pts.push_back({-1, -1});
            }
        }
        prev_hband_x_ = prev_x;

        // ---- No-blob fallback: go straight ----
        bool any_valid = std::any_of(chosen_pts.begin(), chosen_pts.end(),
            [](const std::pair<int, int> &p){ return p.first != -1; });
        if (!any_valid)
            chosen_pts = {{width / 2, height / 2}};

        // ---- Build Path2D poses (horizontal bands) ----
        std::vector<std::pair<int, int>> valid_pts;
        for (auto &p : chosen_pts) if (p.first != -1) valid_pts.push_back(p);

        for (size_t i = 0; i < valid_pts.size(); i++) {
            auto [px, py] = valid_pts[i];
            auto [xr, yr] = image_to_robot(px, py, width, orig_height);

            geometry_msgs::msg::Pose2D pose;
            pose.x = (double)xr;
            pose.y = (double)yr;
            if (i < valid_pts.size() - 1) {
                pose.theta = std::atan2((double)yr, (double)xr);
            } else {
                pose.theta = 0.0;
            }
            path.poses.push_back(pose);
        }

        // ---- Vertical bands (4 bands on right 1/3) ----
        const int num_vert = 4;
        const float vert_bw = std::floor(width * 1.0f / 3.0f / num_vert);
        int prev_vy = -1;

        for (int i = 0; i < num_vert; i++) {
            int x1 = (int)(width * 2.0f / 3.0f + i * vert_bw);
            int x2 = (i == num_vert - 1) ? width
                                          : (int)(width * 2.0f / 3.0f + (i + 1) * vert_bw);
            if (x2 <= x1) x2 = x1 + 1;

            auto blobs = find_blobs(mask_, width, x1, x2, 0, height);

            float best_score = 0.0f;
            int best_cy = -1;

            for (auto &b : blobs) {
                if (b.area < 45) continue;
                int cy = (b.min_y + b.max_y) / 2;

                float y_norm = (cy - orig_height / 2.0f) / (orig_height / 2.0f);
                float y_bias = (y_norm + 1.0f) / 2.0f;
                float cont_p = (prev_vy != -1) ? std::abs(cy - prev_vy) * 2.0f : 0.0f;
                float score  = b.area * 1.0f + y_bias * 2000.0f - cont_p;
                if (y_bias > 0.7f) score += 1500.0f;

                if (score > best_score) {
                    best_score = score;
                    best_cy = cy;
                }
            }

            geometry_msgs::msg::Pose2D pose;
            pose.x = 0.0;
            if (best_cy != -1) {
                pose.y = (double)best_cy;
                pose.theta = 90.0;
            } else {
                pose.y = -99.9;
                pose.theta = std::numeric_limits<double>::infinity();
            }
            path.poses.push_back(pose);
        }

        if (!path.poses.empty())
            path_pub_->publish(path);

        // ---- Latency log (throttled to 1 Hz) ----
        const auto t1 = std::chrono::steady_clock::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "image_listener latency: %.2f ms (%.1f Hz)",
            dt_ms, 1000.0 / std::max(dt_ms, 0.001));
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageListener>());
    rclcpp::shutdown();
    return 0;
}
