from ultralytics import YOLO
import sys
import os
import glob

def resolve_data_path(data_path):
	# look for common dataset YAML filenames first
	for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
		candidate = os.path.join(data_path, name)
		if os.path.isfile(candidate):
			return candidate
	# look for any yaml/yml files at the top level
	yaml_files = glob.glob(os.path.join(data_path, ".yaml")) + glob.glob(os.path.join(data_path, ".yml"))
	if yaml_files:
		return yaml_files[0]
	# search recursively as a last resort
	for root, _, files in os.walk(data_path):
		for f in files:
			if f.endswith((".yaml", ".yml")):
				return os.path.join(root, f)
	# helpful error if nothing is found
	raise FileNotFoundError(
		f"No dataset YAML found in directory '{data_path}'.\n"
		"Provide the path to your dataset .yaml file (e.g. '/path/to/data.yaml') "
		"or add a data.yaml inside the dataset directory (see Ultralytics dataset format)."
	)

def main():
	model_path = "<modeldir>" # Replace <modeldir> with the actual path to your model file
	data_path = "<datdir>"  # Replace <datadir> with the actual path to your dataset directory
	epochs = 50
	imgsz = 640

	try:
		data_path = resolve_data_path(data_path)
		print(f"Using dataset config: {data_path}")
		model = YOLO(model_path)
		# Train the model (see ultralytics docs for additional options)
		model.train(data=data_path, epochs=epochs, imgsz=imgsz)
	except Exception as e:
		print(f"Training failed: {e}", file=sys.stderr)
		sys.exit(1)

if __name__ == "__main__":
	main()