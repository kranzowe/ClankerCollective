#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;

class ImageListener : public rclcpp::Node
{
public:
    ImageListener() : Node("image_listener")
    {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&ImageListener::callback, this, _1));

        bbox_pub_ = this->create_publisher<vision_msgs::msg::BoundingBox2D>(
            "/blob_rectangle", 10);

        path_pub_ = this->create_publisher<nav_2d_msgs::msg::Path2D>(
            "/line_path", 10);

        // HSV parameters
        declare_parameter("fh", 120);
        declare_parameter("fs", 180);
        declare_parameter("fv", 120);

        declare_parameter("th", 45);
        declare_parameter("ts", 80);
        declare_parameter("tv", 100);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<vision_msgs::msg::BoundingBox2D>::SharedPtr bbox_pub_;
    rclcpp::Publisher<nav_2d_msgs::msg::Path2D>::SharedPtr path_pub_;

    // -------- BGR → HSV --------
    void bgr2hsv(uint8_t b, uint8_t g, uint8_t r, int &h, int &s, int &v)
    {
        float bf = b / 255.0f;
        float gf = g / 255.0f;
        float rf = r / 255.0f;

        float cmax = std::max({rf, gf, bf});
        float cmin = std::min({rf, gf, bf});
        float diff = cmax - cmin;

        if (diff == 0) h = 0;
        else if (cmax == rf) h = 60 * fmod(((gf - bf) / diff), 6);
        else if (cmax == gf) h = 60 * (((bf - rf) / diff) + 2);
        else h = 60 * (((rf - gf) / diff) + 4);

        if (h < 0) h += 360;
        h /= 2; // OpenCV scale

        s = (cmax == 0) ? 0 : (diff / cmax) * 255;
        v = cmax * 255;
    }

    int hue_diff(int a, int b)
    {
        int d = abs(a - b);
        return std::min(d, 180 - d);
    }

    // -------- Image → Robot --------
    std::pair<float,float> image_to_robot(int px, int py, int width, int height)
    {
        float x_norm = (px - width / 2.0f) / (width / 2.0f);
        float y_norm = pow((height - py) / (float)height, 1.5);

        float x_robot = y_norm * 1.0;
        float y_robot = x_norm * 1.4;

        return {x_robot, y_robot};
    }

    // -------- Main Callback --------
    void callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (msg->encoding != "bgr8") {
            RCLCPP_WARN(this->get_logger(), "Expected bgr8 image");
            return;
        }

        int width = msg->width;
        int height = msg->height;
        const auto &data = msg->data;

        int fh = get_parameter("fh").as_int();
        int fs = get_parameter("fs").as_int();
        int fv = get_parameter("fv").as_int();

        int th = get_parameter("th").as_int();
        int ts = get_parameter("ts").as_int();
        int tv = get_parameter("tv").as_int();

        std::vector<uint8_t> mask(width * height, 0);

        int min_x = width, min_y = height, max_x = 0, max_y = 0;

        // -------- Build Mask --------
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;

                uint8_t b = data[idx];
                uint8_t g = data[idx+1];
                uint8_t r = data[idx+2];

                int h,s,v;
                bgr2hsv(b,g,r,h,s,v);

                if (hue_diff(h, fh) < th &&
                    abs(s - fs) < ts &&
                    abs(v - fv) < tv)
                {
                    mask[y*width + x] = 1;

                    min_x = std::min(min_x, x);
                    min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x);
                    max_y = std::max(max_y, y);
                }
            }
        }

        // -------- Bounding Box --------
        if (min_x < max_x && min_y < max_y)
        {
            vision_msgs::msg::BoundingBox2D box;
            box.center.position.x = (min_x + max_x)/2.0;
            box.center.position.y = (min_y + max_y)/2.0;
            box.size_x = max_x - min_x;
            box.size_y = max_y - min_y;
            bbox_pub_->publish(box);
        }

        // -------- Path with ORIGINAL LOGIC --------
        nav_2d_msgs::msg::Path2D path;

        int num_bands = 8;
        int band_h = height / num_bands;

        int prev_x = -1;

        for (int i = 0; i < num_bands; i++)
        {
            int y1 = i * band_h;
            int y2 = (i == num_bands-1) ? height : (i+1)*band_h;

            // visited map（avoid duplicate calculation）
            std::vector<uint8_t> visited(width * height, 0);

            struct Blob {
                int area;
                int sum_x;
                int sum_y;
                int min_x, min_y, max_x, max_y;
            };

            std::vector<Blob> blobs;

            // -------- find blobs （replace findContours）--------
            for (int y = y1; y < y2; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;

                    if (mask[idx] == 0 || visited[idx]) continue;

                    // BFS flood fill
                    std::vector<std::pair<int,int>> queue;
                    queue.push_back({x,y});
                    visited[idx] = 1;

                    Blob blob;
                    blob.area = 0;
                    blob.sum_x = 0;
                    blob.sum_y = 0;
                    blob.min_x = x;
                    blob.max_x = x;
                    blob.min_y = y;
                    blob.max_y = y;

                    for (size_t q = 0; q < queue.size(); q++)
                    {
                        auto [cx, cy] = queue[q];
                        int cidx = cy * width + cx;

                        blob.area++;
                        blob.sum_x += cx;
                        blob.sum_y += cy;

                        blob.min_x = std::min(blob.min_x, cx);
                        blob.max_x = std::max(blob.max_x, cx);
                        blob.min_y = std::min(blob.min_y, cy);
                        blob.max_y = std::max(blob.max_y, cy);

                        // 4-neighbor
                        int dx[4] = {1,-1,0,0};
                        int dy[4] = {0,0,1,-1};

                        for (int k = 0; k < 4; k++)
                        {
                            int nx = cx + dx[k];
                            int ny = cy + dy[k];

                            if (nx < 0 || nx >= width || ny < y1 || ny >= y2) continue;

                            int nidx = ny * width + nx;

                            if (mask[nidx] && !visited[nidx])
                            {
                                visited[nidx] = 1;
                                queue.push_back({nx,ny});
                            }
                        }
                    }

                    blobs.push_back(blob);
                }
            }

            if (blobs.empty()) continue;

            // -------- SCORING（perfect following Python）--------
            float best_score = -1e9;
            Blob best_blob;

            for (auto &b : blobs)
            {
                if (b.area < 45) continue;  // same threshold

                int cx = b.sum_x / b.area;
                int cy = b.sum_y / b.area;

                // normalize x
                float x_norm = (cx - width / 2.0f) / (width / 2.0f);
                float x_bias = (x_norm + 1.0f) / 2.0f;

                float continuity_penalty = 0;
                if (prev_x != -1)
                {
                    continuity_penalty = abs(cx - prev_x) * 2.0;
                }

                float area_weight = 1.0;
                float right_weight = 2000.0;

                float score = (b.area * area_weight)
                            + (x_bias * right_weight)
                            - continuity_penalty;

                if (x_bias > 0.7)
                {
                    score += 1500.0;
                }

                if (score > best_score)
                {
                    best_score = score;
                    best_blob = b;
                }
            }

            // -------- pick best blob --------
            int cx = best_blob.sum_x / best_blob.area;
            int cy = best_blob.sum_y / best_blob.area;

            prev_x = cx;

            auto [xr, yr] = image_to_robot(cx, cy, width, height);

            geometry_msgs::msg::Pose2D pose;
            pose.x = xr;
            pose.y = yr;
            pose.theta = 0.0;

            path.poses.push_back(pose);
        }

        if (!path.poses.empty())
        {
            path_pub_->publish(path);
        }
        
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageListener>());
    rclcpp::shutdown();
    return 0;
}