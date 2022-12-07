import fiftyone as fo
import os

gt_path = "iray/labels/"
image_path = "iray/images/"

gt_files = os.listdir(gt_path)

dataset = fo.Dataset()
# Set default classes
dataset.default_classes = ["Person", "Vehicle", "Animal", "Bike"]
dataset.save()  # must save after edits

# Create samples for your data
samples = []
for i in range(len(gt_files)):
#for i in range(1):
    image_name = image_path+gt_files[i]
    image_name = image_name[:-3] + 'png'
    print(str(i + 1) + ' : ' + image_name)

    gt_file = open(gt_path+gt_files[i])
    sample = fo.Sample(filepath=image_name)
    detections = []

    for line_text  in gt_file:
        tokens = line_text.split()

        label = dataset.default_classes[int(tokens[0])]

        box_width = float(tokens[3])
        box_height = float(tokens[4])
        left_top_x = float(tokens[1]) - box_width/2
        left_top_y = float(tokens[2]) - box_height/2
        bounding_box = [left_top_x, left_top_y, box_width, box_height]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )
        # Store detections in a field name of your choice
        sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)

# Create dataset
print('add_samples...')
dataset.add_samples(samples)
dataset.save()  # must save after edits

print('export...')
# Export the dataset
dataset.export(
    export_dir="iray/",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)