# Configuration file for polygon generation

# === contours - contours_img.py ===
path_image_prediction = r"img_prediction_join"  # for test, for production need to use folder after prediction join postprocessing

path_image_contours = r"img_contoures"
path_polygons_shp = r"polygons_shp"
min_area = 500             # Minimum polygon area for noise filtering
contour_thickness = 3      # Red contour thickness