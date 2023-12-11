import xml.etree.ElementTree as ET
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # get id and name of each of the images
    for image_tag in root.findall(".//image"):
        image_info = {
            'id': image_tag.get('id'),
            'name': image_tag.get('name'),
        }

        # get info for polygons in each image if exists
        for poly_tag in image_tag.findall('.//polygon'):
            poly_info = {
                'label': poly_tag.get('label'),
                'points': poly_tag.get('points'),
            }
            if 'polygons' not in image_info:
                image_info['polygons'] = []
            image_info['polygons'].append(poly_info)

        # get info for masks in each image if exists
        for mask_tag in image_tag.findall('.//mask'):
            mask_info = {
                'label': mask_tag.get('label'),
                'rle': [int(i) for i in mask_tag.get('rle').split(',')],  # convert to list of int from str
                'left': int(mask_tag.get('left')),
                'top': int(mask_tag.get('top')),
                'width': int(mask_tag.get('width')),
                'height': int(mask_tag.get('height')),
            }
            image_info.setdefault('masks', []).append(mask_info)

        annotations.append(image_info)

    return annotations


def create_label_matrix(annotations, save_loc):
    # label_image = []
    for image_info in annotations:
        name = image_info['name']
        print(f"Creating labels for Image {image_info['id']} - {name}")

        # default width and height of the original images collected are 2000 and 900 respectively
        width = 2000
        height = 900

        masked_image = np.array(np.zeros([height, width]), dtype=np.uint8)
        # masks are the potato blocks in these annotations
        if 'masks' in image_info:
            print(f"    Number of Mask: {len(image_info.get('masks'))}")
            for mask in image_info.get('masks'):
                print(f"Mask Label: {mask['label']}")
                rle, left, top, target_width, target_height = (mask['rle'], mask['left'], mask['top'], mask['width'],
                                                               mask['height'])
                if mask['label'] == 'potato_block':
                    tmp_masked_image = rle2mask(rle, width, height, left, top, target_width, target_height, 1)
                    masked_image = np.maximum(masked_image, tmp_masked_image)
                elif mask['label'] == 'resistant':
                    tmp_masked_image = rle2mask(rle, width, height, left, top, target_width, target_height, 2)
                    masked_image = np.maximum(masked_image, tmp_masked_image)

        poly_image = np.array(np.zeros([height, width]), dtype=np.uint8)

        if 'polygons' in image_info:
            print(f"    Number of Polygons: {len(image_info.get('polygons'))}")
            for poly in image_info.get('polygons'):
                print(f"Polygon Label: {poly['label']}")
                points = poly['points']
                # points is in str, and need to converted to coordinate system
                points = points.split(';')
                points = [tuple(map(float, pair.split(','))) for pair in points]

                if poly['label'] == 'pvy_negative':
                    tmp_poly_image = poly2mask(points, width, height, 3)
                    poly_image = np.maximum(poly_image, tmp_poly_image)

                if poly['label'] == 'pvy_positive':
                    tmp_poly_image = poly2mask(points, width, height, 4)
                    poly_image = np.maximum(poly_image, tmp_poly_image)

                if poly['label'] == 'unknown':
                    tmp_poly_image = poly2mask(points, width, height, 5)
                    poly_image = np.maximum(poly_image, tmp_poly_image)

        label_image = np.multiply(masked_image, poly_image)
        print(f"Maximum value: {np.max(label_image)}, Unique Values: {np.unique(label_image)}")
        save_file = os.path.join(save_loc, name + '.npy')
        np.save(str(save_file), label_image)

        # display/save the image
        # labels ww will have: 0, 3, 4, 5, 6, 10

        plt.imshow(label_image, cmap='viridis')
        plt.title('Label for ' + name)
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Label Values')
        save_img = os.path.join(save_loc, name)
        plt.savefig(save_img, dpi=300)
        plt.close()
        # plt.show()


if __name__ == "__main__":
    xml_file_path = os.path.join(info()['general_dir'], 'annotations.xml')
    annotations_data = parse_annotations(xml_file_path)
    save_dir = os.path.join(info()['save_dir'], 'labels')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    create_label_matrix(annotations_data, save_dir)
    print('Done!')
    # print(annotations_data)

    # for image in annotations_data:
    #     print(f"Image ID: {image['id']}, Image Name: {image['name']}")
    #     for poly in image.get('polygons', []):
    #         print(f"  Label: {poly.get('label')}, "
    #               f"Coordinates: ({poly.get('points')})")
