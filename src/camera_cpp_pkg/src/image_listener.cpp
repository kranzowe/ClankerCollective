#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using std::placeholders::_1;

// ---- BGR → HLS_FULL  (H, L, S each in [0, 255]) ----
static void bgr2hls_full(uint8_t b, uint8_t g, uint8_t r,
                          int &H, int &L, int &S)
{
    float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
    float cmax = std::max({rf, gf, bf});
    float cmin = std::min({rf, gf, bf});
    float diff = cmax - cmin;
    float lf   = (cmax + cmin) * 0.5f;

    float sf = 0.0f;
    if (diff > 1e-6f)
        sf = (lf <= 0.5f) ? diff / (cmax + cmin) : diff / (2.0f - cmax - cmin);

    float hf = 0.0f;
    if (diff > 1e-6f) {
        if      (cmax == rf) hf = 60.0f * fmodf((gf - bf) / diff, 6.0f);
        else if (cmax == gf) hf = 60.0f * ((bf - rf) / diff + 2.0f);
        else                 hf = 60.0f * ((rf - gf) / diff + 4.0f);
        if (hf < 0.0f) hf += 360.0f;
    }

    H = std::clamp((int)(hf * 255.0f / 360.0f + 0.5f), 0, 255);
    L = std::clamp((int)(lf * 255.0f + 0.5f),          0, 255);
    S = std::clamp((int)(sf * 255.0f + 0.5f),          0, 255);
}

// ---- 5×5 Gaussian blur  [1,4,6,4,1]/16 separable ----
// Two-pass: horizontal accumulates without dividing, vertical divides by 256 total.
static void gaussian_blur_bgr(const uint8_t *src, std::vector<uint8_t> &dst, int w, int h)
{
    static const int K[5] = {1, 4, 6, 4, 1};
    dst.assign(w * h * 3, 0);
    std::vector<int> tmp(w * h);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int sum = 0;
                for (int k = -2; k <= 2; k++)
                    sum += src[(y * w + std::clamp(x + k, 0, w - 1)) * 3 + c] * K[k + 2];
                tmp[y * w + x] = sum;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int sum = 0;
                for (int k = -2; k <= 2; k++)
                    sum += tmp[std::clamp(y + k, 0, h - 1) * w + x] * K[k + 2];
                dst[(y * w + x) * 3 + c] = (uint8_t)std::clamp(sum / 256, 0, 255);
            }
    }
}

// ---- Morphological helpers (5×5 rect kernel, via integral image) ----
static std::vector<int> build_ii(const std::vector<uint8_t> &mask, int w, int h)
{
    std::vector<int> ii((w + 1) * (h + 1), 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            ii[(y+1)*(w+1)+(x+1)] = mask[y*w+x]
                + ii[y*(w+1)+(x+1)] + ii[(y+1)*(w+1)+x] - ii[y*(w+1)+x];
    return ii;
}

static inline int ii_sum(const std::vector<int> &ii, int W1,
                          int y1, int x1, int y2, int x2)
{
    return ii[y2*W1+x2] - ii[y1*W1+x2] - ii[y2*W1+x1] + ii[y1*W1+x1];
}

static void erode5(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int w, int h)
{
    auto ii = build_ii(src, w, h); int W1 = w + 1;
    dst.resize(w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int y1=std::max(0,y-2), y2=std::min(h,y+3);
            int x1=std::max(0,x-2), x2=std::min(w,x+3);
            dst[y*w+x] = (ii_sum(ii,W1,y1,x1,y2,x2)==(y2-y1)*(x2-x1)) ? 1 : 0;
        }
}

static void dilate5(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int w, int h)
{
    auto ii = build_ii(src, w, h); int W1 = w + 1;
    dst.resize(w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int y1=std::max(0,y-2), y2=std::min(h,y+3);
            int x1=std::max(0,x-2), x2=std::min(w,x+3);
            dst[y*w+x] = (ii_sum(ii,W1,y1,x1,y2,x2) > 0) ? 1 : 0;
        }
}

// ================================================================
class ImageListener : public rclcpp::Node
{
public:
    ImageListener() : Node("image_listener")
    {
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&ImageListener::callback, this, _1));

        bbox_pub_ = create_publisher<vision_msgs::msg::BoundingBox2D>("/blob_rectangle", 10);
        path_pub_ = create_publisher<nav_2d_msgs::msg::Path2D>("/line_path", 10);

        declare_parameter("fh", 120);
        declare_parameter("fs", 180);
        declare_parameter("fl", 120);   // lightness center  (was fv / Value)

        declare_parameter("th", 45);
        declare_parameter("ts", 80);
        declare_parameter("tl", 100);   // lightness tolerance (was tv)
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<vision_msgs::msg::BoundingBox2D>::SharedPtr bbox_pub_;
    rclcpp::Publisher<nav_2d_msgs::msg::Path2D>::SharedPtr path_pub_;

    int prev_hband_x_ = -1;   // cross-frame continuity for horizontal bands

    // ---- Image → Robot (same as Python) ----
    std::pair<float,float> image_to_robot(int px, int py, int width, int height)
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
        std::vector<uint8_t> vis(rw * rh, 0);

        for (int y = ys; y < ye; y++) {
            for (int x = xs; x < xe; x++) {
                int lx = x - xs, ly = y - ys;
                if (!mask[y * full_w + x] || vis[ly * rw + lx]) continue;

                std::vector<std::pair<int,int>> q;
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

                    const int dx[4] = {1,-1,0,0};
                    const int dy[4] = {0,0,1,-1};
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
        if (msg->encoding != "bgr8") {
            RCLCPP_WARN(get_logger(), "Expected bgr8, got %s", msg->encoding.c_str());
            return;
        }

        const int orig_height = (int)msg->height;
        const int width       = (int)msg->width;

        // ---- Crop top 25% (matching Python crop_ratio=0.75) ----
        int start_y = (int)(orig_height * 0.25);
        int height  = orig_height - start_y;   // cropped frame height
        const uint8_t *crop_ptr = msg->data.data() + start_y * width * 3;

        // ---- Gaussian blur 5×5 ----
        std::vector<uint8_t> blurred;
        gaussian_blur_bgr(crop_ptr, blurred, width, height);

        // ---- Parameters ----
        int fh = get_parameter("fh").as_int();
        int fs = get_parameter("fs").as_int();
        int fl = get_parameter("fl").as_int();
        int th = get_parameter("th").as_int();
        int ts = get_parameter("ts").as_int();
        int tl = get_parameter("tl").as_int();

        int lo_h = fh - th, hi_h = fh + th;
        int lo_l = fl - tl, hi_l = fl + tl;
        int lo_s = fs - ts, hi_s = fs + ts;

        // ---- Build HLS_FULL mask ----
        std::vector<uint8_t> mask_raw(width * height, 0);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                int H, L, S;
                bgr2hls_full(blurred[idx], blurred[idx+1], blurred[idx+2], H, L, S);
                if (H >= lo_h && H <= hi_h &&
                    L >= lo_l && L <= hi_l &&
                    S >= lo_s && S <= hi_s)
                    mask_raw[y * width + x] = 1;
            }
        }

        // ---- Morphological OPEN (erode→dilate) then CLOSE (dilate→erode) ----
        std::vector<uint8_t> tmp, mask;
        erode5 (mask_raw, tmp,  width, height);
        dilate5(tmp,  mask_raw, width, height);   // OPEN  → mask_raw
        dilate5(mask_raw, tmp,  width, height);
        erode5 (tmp,  mask,     width, height);   // CLOSE → mask

        // ---- Bounding box: largest blob with area > 500 ----
        {
            auto blobs = find_blobs(mask, width, 0, width, 0, height);
            if (!blobs.empty()) {
                auto &best = *std::max_element(blobs.begin(), blobs.end(),
                    [](const Blob &a, const Blob &b){ return a.area < b.area; });
                if (best.area > 500) {
                    vision_msgs::msg::BoundingBox2D box;
                    box.center.position.x = (best.min_x + best.max_x) / 2.0;
                    box.center.position.y = (best.min_y + best.max_y) / 2.0;
                    box.size_x = best.max_x - best.min_x;
                    box.size_y = best.max_y - best.min_y;
                    bbox_pub_->publish(box);
                }
            }
        }

        // ---- Horizontal bands (8 bands, orig_height for band_h — matches Python) ----
        nav_2d_msgs::msg::Path2D path;
        const int num_bands = 8;
        int band_h = orig_height / num_bands;
        int prev_x = prev_hband_x_;
        std::vector<std::pair<int,int>> chosen_pts;

        for (int i = 0; i < num_bands; i++) {
            int y1 = i * band_h;
            int y2 = (i == num_bands - 1) ? orig_height : (i + 1) * band_h;
            if (y1 >= height) break;
            y2 = std::min(y2, height);

            auto blobs = find_blobs(mask, width, 0, width, y1, y2);

            float best_score = 0.0f;   // init 0 matching Python's best_area=0
            int best_cx = -1, best_cy = -1;

            for (auto &b : blobs) {
                if (b.area < 45) continue;
                int cx = (b.min_x + b.max_x) / 2;
                int cy = (b.min_y + b.max_y) / 2;

                float x_norm  = (cx - width / 2.0f) / (width / 2.0f);
                float x_bias  = (x_norm + 1.0f) / 2.0f;
                float cont_p  = (prev_x != -1) ? std::abs(cx - prev_x) * 2.0f : 0.0f;
                float score   = b.area * 1.0f + x_bias * 2000.0f - cont_p;
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
            [](const std::pair<int,int> &p){ return p.first != -1; });
        if (!any_valid)
            chosen_pts = {{width / 2, height / 2}};

        // ---- Build Path2D poses (horizontal bands) ----
        std::vector<std::pair<int,int>> valid_pts;
        for (auto &p : chosen_pts) if (p.first != -1) valid_pts.push_back(p);

        for (size_t i = 0; i < valid_pts.size(); i++) {
            auto [px, py] = valid_pts[i];
            // use orig_height for normalization — matches Python (height stored before crop)
            auto [xr, yr] = image_to_robot(px, py, width, orig_height);

            geometry_msgs::msg::Pose2D pose;
            pose.x = (double)xr;
            pose.y = (double)yr;
            // theta: atan2 of current robot point (matches Python — next_pt computed but unused)
            pose.theta = (i < valid_pts.size() - 1) ? std::atan2((double)yr, (double)xr) : 0.0;
            path.poses.push_back(pose);
        }

        // ---- Vertical bands (4 bands on right 1/3) ----
        const int num_vert = 4;
        // floor division matches Python's (width*1.0/3.0) // num_vert_bands
        float vert_bw = std::floor(width * 1.0f / 3.0f / num_vert);
        int prev_vy = -1;   // never updated in loop — matches Python (prev_point not updated)

        for (int i = 0; i < num_vert; i++) {
            int x1 = (int)(width * 2.0f / 3.0f + i * vert_bw);
            int x2 = (i == num_vert - 1) ? width
                                          : (int)(width * 2.0f / 3.0f + (i + 1) * vert_bw);
            if (x2 <= x1) x2 = x1 + 1;

            auto blobs = find_blobs(mask, width, x1, x2, 0, height);

            float best_score = 0.0f;
            int best_cy = -1;

            for (auto &b : blobs) {
                if (b.area < 45) continue;
                int cy = (b.min_y + b.max_y) / 2;

                // use orig_height — matches Python (height = original before crop)
                float y_norm  = (cy - orig_height / 2.0f) / (orig_height / 2.0f);
                float y_bias  = (y_norm + 1.0f) / 2.0f;
                float cont_p  = (prev_vy != -1) ? std::abs(cy - prev_vy) * 2.0f : 0.0f;
                float score   = b.area * 1.0f + y_bias * 2000.0f - cont_p;
                if (y_bias > 0.7f) score += 1500.0f;

                if (score > best_score) {
                    best_score = score;
                    best_cy = cy;
                }
            }

            geometry_msgs::msg::Pose2D pose;
            pose.x = 0.0;   // matches Python: pose.x = float() = 0.0
            if (best_cy != -1) {
                pose.y     = (double)best_cy;  // raw pixel y (matches Python)
                pose.theta = 90.0;
            } else {
                pose.y     = -99.9;
                pose.theta = std::numeric_limits<double>::infinity();
            }
            path.poses.push_back(pose);
        }

        if (!path.poses.empty())
            path_pub_->publish(path);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageListener>());
    rclcpp::shutdown();
    return 0;
}
