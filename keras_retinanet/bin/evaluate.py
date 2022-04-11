#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from ..utils.eval import evaluate
from ..utils.gpu import setup_gpu
from ..utils.tf_version import check_tf_version
from ..utils.F1_score_precision_recall import calculate_F1_precision_recall, plot_each_species, plot_F1_precision_recall


def create_generator(args, preprocess_image):
    """ Create generators for evaluation.
    """
    common_args = {
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
        'group_method'     : args.group_method
    }

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_extension=args.image_extension,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--image-extension',   help='Declares the dataset images\' extension.', default='.jpg')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--no-resize',        help='Don''t rescale the image.', action='store_true')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--missed_detections_path', help='Path for saving images with missed detections.')
    parser.add_argument('--group-method',     help='Determines how images are grouped together', type=str, default='ratio', choices=['none', 'random', 'ratio'])
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure tensorflow is the minimum required version
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    backbone = models.backbone(args.backbone)
    generator = create_generator(args, backbone.preprocess_image)

    # optionally load anchor parameters
    anchor_params = None
    pyramid_levels = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
    if args.config and 'pyramid_levels' in args.config:
        pyramid_levels = parse_pyramid_levels(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)
    generator.compute_shapes = make_shapes_callback(model)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, class_specific_filter=False,
                                     classanchor_params=anchor_params,
                                     pyramid_levels=pyramid_levels)
    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        average_precisions, all_scores, all_true_positives, inference_time = evaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path,
            missed_detections_path=args.missed_detections_path
        )
        # print evaluation
        total_instances = []
        precisions = []

        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label),
                  'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        ### Calculation of F1_score, precision and recall ####

        num_classes = generator.num_classes()
        label_dict = {label: generator.label_to_name(label) for label in range(num_classes)}

        if sys.argv[1:][0] == "--score-threshold":
            F1_score, precision, recall = calculate_F1_precision_recall(args.score_threshold,
                                                                        num_classes,
                                                                        total_instances,
                                                                        all_scores,
                                                                        all_true_positives)
            label_dict[num_classes] = "all_species"
            print("F1 score:",
                  [label_dict[label] + ": {0:.3f}".format(F1_score[label]) for label in label_dict])

        else:
            score_range = np.arange(args.score_threshold, 1, 0.01)
            precision_list = [[] for x in range(num_classes + 1)]
            recall_list = [[] for x in range(num_classes + 1)]
            F1_score_list = [[] for x in range(num_classes + 1)]

            for thr in score_range:
                F1_score, precision, recall = calculate_F1_precision_recall(thr, num_classes,
                                                                            total_instances,
                                                                            all_scores,
                                                                            all_true_positives)
                for label in range(num_classes+1):
                    F1_score_list[label].append(F1_score[label])
                    precision_list[label].append(precision[label])
                    recall_list[label].append(recall[label])

            opt_score_thres = [score_range[np.argmax(i)] for label, i in enumerate(F1_score_list)]

            # plot_each_species(score_range, precision_list, label_dict, "Precision", args.annotations)
            # plot_each_species(score_range, recall_list, label_dict, "Recall", args.annotations)
            # plot_each_species(score_range, F1_score_list, label_dict, "F1_score", args.annotations, opt_score_thres)
            # plot_F1_precision_recall(score_range, F1_score_list, precision_list, recall_list, num_classes,
            #                          args.annotations)

            label_dict[num_classes] = "all_species"
            print("Optimal_score_threshold:", [label_dict[label] +
                                           ": {0:.3f}".format(score_range[np.argmax(i)])
                                           for label, i in enumerate(F1_score_list)])
            print("Max F1 score:",
                  [label_dict[label] + ": {0:.3f}".format(np.max(F1_score_list[label])) for label in label_dict])

        print('mAP using the weighted average of precisions among classes: {:.4f}'.
               format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        print('Inference time for {:.0f} images: {:.4f}'.format(generator.size(), inference_time))


if __name__ == '__main__':
    main()
