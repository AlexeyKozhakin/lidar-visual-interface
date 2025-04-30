

# === contours - countours_img.py ===
path_image_prediction = r"temp\img_prediction_join" # for test, for production need to use folder afte predicton join postropcessing


path_image_contours = r"temp\img_contoures"
path_polygons_shp = r"temp\polygons_shp"
min_area = 500             # ✅ Минимальная площадь полигона для фильтрации шумов
contour_thickness = 3      # Толщина красного контура