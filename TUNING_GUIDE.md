# HLS Thresholding & Waypoint Tuning Guide

## Problem: Only Getting 1 Waypoint at (0.35, 0, 0)

This means **no valid blobs are being detected in any band**. The node falls back to the center point.

## Quick Diagnosis (5 minutes)

### Step 1: Enable Debug Visualizations
```bash
# In a new terminal, set debug mode to true
ros2 param set /image_listener enable_debug_viz true

# View debug topics in real-time
ros2 topic echo /debug/mask_raw  # Should show white pixels where line is detected
ros2 topic echo /debug/hls_frame # Should show line in HLS color space
```

### Step 2: Check Debug Images in RViz
```bash
# Launch RViz and subscribe to these topics:
- /debug/original_cropped   # What the node sees (bottom 25% of image)
- /debug/hls_frame         # HLS color space representation
- /debug/mask_raw          # Binary mask after thresholding (white=detected, black=background)
- /debug/mask_morphology   # After morphology filtering
- /debug/bands_and_waypoints  # With band divisions and waypoints overlaid
```

### Step 3: Read the Logs
```bash
# Run node with debug logging enabled
export ROS_LOG_LEVEL=DEBUG
ros2 run camera_test_pkg camera_test_pkg my_color_sub.py

# Watch for these lines:
# "Mask pixels: XXX (Y% of frame)"  ← Should be >1% if line is visible
# "Band extraction: N/8 bands found valid blobs"  ← Should be >0
# "HLS Thresholds Updated: ..."  ← Shows current detection bounds
```

## Common Causes & Fixes

### Issue: No pixels detected (`mask_raw` is all black)

**Cause 1: HLS bounds are too strict**
- Current defaults: H: 70–150, L: 40–140, S: 80–220

**Fix:**
```bash
# Widen the bounds incrementally
ros2 param set /image_listener th 60   # Increase hue tolerance (was 40)
ros2 param set /image_listener tl 70   # Increase lightness tolerance (was 50)
ros2 param set /image_listener ts 100  # Increase saturation tolerance (was 70)
```

**Cause 2: Dark blue isn't centered correctly**
- The tape might be darker or more saturated than expected

**Fix:**
```bash
# Interactively sample: use Image viewer to pick pixel values
# Then convert BGR pixel to HLS and update center values:

# Example: if you sample a dark blue pixel in your image,
# look at its BGR values, convert to HLS [0-255] scale

ros2 param set /image_listener fh 100   # Try lower hue
ros2 param set /image_listener fl 80    # Try lower lightness (darker tape)
ros2 param set /image_listener fs 130   # Adjust saturation center
```

### Issue: Too many false positives (non-line regions detected)

**Fix:**
- Use `/debug/bands_and_waypoints` to see which regions are being selected
- Tighten the HLS bounds:

```bash
ros2 param set /image_listener th 25   # Decrease hue tolerance
ros2 param set /image_listener tl 30   # Decrease lightness tolerance
```

### Issue: Line is detected in some bands, not others (inconsistent)

**Cause:** Lighting variations or shadows across the image

**Fix:**
- Increase tolerances slightly (0.5–1.0× current)
- Or increase min_blob_area if you're detecting noise:

```bash
ros2 param set /image_listener min_blob_area 50
```

## Advanced Tuning

### Waypoint Smoothing (if they jump around)
```bash
ros2 param set /image_listener smooth_factor 0.15  # Less smoothing (default 0.25)
ros2 param set /image_listener smooth_factor 0.35  # More smoothing (default 0.25)
```

### Theta (Heading) Fitting Window
```bash
ros2 param set /image_listener fit_window 5   # Fit to more points (smoother heading)
ros2 param set /image_listener fit_window 1   # Responsive heading (default 3)
```

### Image Processing
```bash
ros2 param set /image_listener blur_ksize 7      # Stronger blur (default 5)
ros2 param set /image_listener morph_ksize 7     # Larger morphology kernel
ros2 param set /image_listener num_bands 12      # More waypoints (default 8)
```

## Recommended Tuning Workflow

1. **Start with debug images running**
   - View `/debug/original_cropped` + `/debug/mask_raw` side-by-side
   
2. **Adjust HLS center values** (fh, fl, fs)
   - Increase bounds until you see the line in `mask_raw`
   
3. **Decrease false positives** (if many)
   - Tighten tolerances (th, tl, ts)
   - Increase min_blob_area
   
4. **Test waypoint quality**
   - `ros2 topic echo /line_path`
   - Should see 8+ waypoints, smoothly changing across frames
   
5. **Fine-tune smoothing** if needed
   - Test 90° turns at different smooth_factor values

## HLS Reference for Dark Blue Painters Tape

**Typical ranges [0-255 scale]:**
- **Hue**: 100–120 (dark blue, not light blue)
- **Lightness**: 70–110 (not too dark, not light)
- **Saturation**: 100–180 (reasonably saturated)

**Starting bounds (if defaults don't work):**
```bash
ros2 param set /image_listener fh 110   # Hue center
ros2 param set /image_listener fl 90    # Lightness center
ros2 param set /image_listener fs 140   # Saturation center

ros2 param set /image_listener th 50    # Hue tolerance
ros2 param set /image_listener tl 60    # Lightness tolerance
ros2 param set /image_listener ts 90    # Saturation tolerance
```

## Save Working Configuration

Once tuned, save the parameters:
```bash
# Dump current configuration
ros2 param dump /image_listener > line_detector_params.yaml

# Later, reload:
ros2 param load /image_listener line_detector_params.yaml
```

## Still Not Working?

Check these in order:
1. [ ] Is `/debug/original_cropped` showing the cropped image? (bottom 25% of frame)
2. [ ] Is the line visible in that cropped region?
3. [ ] What's the log output for "Mask pixels: X%"? (should be >1%)
4. [ ] Try viewing raw camera in `rqt_image_view` to confirm line is visible at all

If the line is visible in the camera but not detected:
- The HLS bounds need adjustment
- Post the "Mask pixels" percentage and we can help tune further
