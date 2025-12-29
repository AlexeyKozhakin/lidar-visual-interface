# === prediction settings
from importlib.resources import files


checkpoint_path = files("musac_las_classifier.predictor_multiclass_segmentation") / "model/model_epoch_31.pth"
encoder_weights_path =  files("musac_las_classifier.predictor_multiclass_segmentation") / "model/resnet34-333f7ec4.pth"
output_directory = r'example_data/img_predict_multicalss'
output_directory_join = "temp/img_predict_multi_class_join"