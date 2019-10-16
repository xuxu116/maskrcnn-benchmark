import math

import torch
import torch.nn.functional as F
from torch import nn

#from maskrcnn_benchmark.modeling.box_coder import BoxCoder
#from maskrcnn_benchmark.layers import Conv2d_dw
from maskrcnn_benchmark.modeling.rpn.free_anchor_loss import make_free_anchor_loss_evaluator

from ..anchor_generator import make_anchor_generator_retinanet
from .inference import make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        #conv_block = Conv2d_dw if cfg.MODEL.RETINANET.DEPTHWISE else nn.Conv2d
        conv_block = nn.Conv2d
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        cls_tower = []
        bbox_tower = []

        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                conv_block(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        out_channels =  in_channels
        self.cls_logits = conv_block(
            out_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = conv_block(
            out_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower,  self.cls_logits,
                  self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if (type(l.bias)!=type(None)):
                        torch.nn.init.constant_(l.bias, 0)


        # retinanet_bias_init
        if isinstance(self.cls_logits, nn.Conv2d):
            prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            f = self.cls_tower(feature)
            logits.append(self.cls_logits(f))
            bbox_reg.append(self.bbox_pred(f))
            #logits.append(self.cls_logits(self.cls_tower(feature)))
            #bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        if cfg.FREEANCHOR.FREEANCHOR_ON:
            self.loss_name = ["loss_recall", "loss_precision"]
            from maskrcnn_benchmark.modeling.box_coder_free_anchor import BoxCoder
        else:
            self.loss_name = ["loss_retina_cls", "loss_retina_reg"]
            from maskrcnn_benchmark.modeling.box_coder import BoxCoder
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator =make_free_anchor_loss_evaluator(cfg, box_coder) if cfg.FREEANCHOR.FREEANCHOR_ON \
            else make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            self.loss_name[0]: loss_box_cls,
            self.loss_name[1]: loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
