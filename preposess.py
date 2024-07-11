import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

def crop_and_save_face(image_path, output_dir, confidence_threshold=0.8, extend_crop_percentage=0.0, min_img_size=128):
    image = cv2.imread(image_path)
    if image is None:
        return [False, "input_error"]
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results  = detector.detect_faces(image_rgb)
    junk_folder_path = os.path.join('datasets', 'junk', output_dir)

    if len(results) == 0:
        save_to_junk(image_rgb, junk_folder_path, "NoFaceDetect", output_dir, image_path)
        return [False, "no_face_detected"]

    most_confident_face = max(results, key=lambda r: r['confidence']) if results else None
    if not most_confident_face or most_confident_face['confidence'] < confidence_threshold:
        save_to_junk(image_rgb, junk_folder_path, "BelowThreshold", output_dir, image_path)
        return [False, "below_threshold"]

    x, y, width, height = abs(most_confident_face['box'][0]), abs(most_confident_face['box'][1]), most_confident_face['box'][2], most_confident_face['box'][3]
    extend_x, extend_y = int(width * extend_crop_percentage), int(height * extend_crop_percentage)
    x_start, y_start = max(0, x - extend_x), max(0, y - extend_y)
    x_end, y_end = min(image_rgb.shape[1], x + width + extend_x), min(image_rgb.shape[0], y + height + extend_y)
    face = image_rgb[y_start:y_end, x_start:x_end]

    if face.shape[1] < min_img_size or face.shape[0] < min_img_size:
        save_to_junk(image_rgb, junk_folder_path, "LessThanMinSize", output_dir, image_path)
        return [False, "small"]

    save_cropped_face(face, output_dir, image_path)
    return [True, "success"]

def save_to_junk(image, junk_folder_path, prefix, output_dir, image_path):
    os.makedirs(junk_folder_path, exist_ok=True)
    junk_output_path = f"{junk_folder_path}/{prefix}_{output_dir}_{os.path.basename(image_path).split('.')[0]}.jpg"
    plt.imsave(junk_output_path, image)

def save_cropped_face(face, output_dir, image_path):
    output_folder_path = os.path.join('datasets', 'FAS_preprocess', output_dir, os.path.basename(os.path.dirname(image_path)))
    os.makedirs(output_folder_path, exist_ok=True)
    face_output_path = f"{output_folder_path}/{os.path.basename(image_path).split('.')[0]}.jpg"
    plt.imsave(face_output_path, face)

def process_image(args, folder_path, folder_target, file_path):
    full_image_path = os.path.join(folder_path, folder_target, file_path)
    return crop_and_save_face(full_image_path, folder_target, args.confidence, args.extend_crop, args.min_img_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extend_crop', type=float, default=0., help='extend_crop_percentage')
    parser.add_argument('--confidence', type=float, default=0.8, help='confidence_threshold')
    parser.add_argument('--min_img_size', type=int, default=128, help='min image size')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    folder_path = "datasets/FAS"
    folder_contents = os.listdir(folder_path)
    summary_text = ""
    for folder_target in folder_contents:
        df = pd.read_csv(os.path.join(folder_path, folder_target, 'label.csv'))
        results = Parallel(n_jobs=4)(delayed(process_image)(args, folder_path, folder_target, df.iloc[i, 0]) for i in tqdm(range(len(df)), desc="Processing images"))

        new_df = pd.DataFrame()
        image_found_number = sum(1 for result in results if result[0])
        error_small_image_number = sum(1 for result in results if result[1] == "small")
        error_not_found_number = sum(1 for result in results if result[1] == "no_face_detected")
        error_below_threshold_number = sum(1 for result in results if result[1] == "below_threshold")

        new_df = pd.concat([df.iloc[[i]] for i, result in enumerate(results) if result[0]], ignore_index=True)
        output_folder_path = os.path.join('datasets', 'FAS_preprocess', folder_target)
        os.makedirs(output_folder_path, exist_ok=True)
        new_df.to_csv(os.path.join(output_folder_path, 'label.csv'), index=False)

        total_images = len(df)
        
        print(f"Finished processing images. Found: {(image_found_number/total_images)*100:.2f}% | Small: {(error_small_image_number/total_images)*100:.2f}% | Not found: {(error_not_found_number/total_images)*100:.2f}% | Below threshold: {(error_below_threshold_number/total_images)*100:.2f}%")
        summary_text += f"Folder {folder_target} Finished. Found: {(image_found_number/total_images)*100:.2f}% | Small: {(error_small_image_number/total_images)*100:.2f}% | Not found: {(error_not_found_number/total_images)*100:.2f}% | Below threshold: {(error_below_threshold_number/total_images)*100:.2f}%" + "\n"
    print(summary_text)