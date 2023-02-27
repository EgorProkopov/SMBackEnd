import cv2


def shape_content_tps_warp(clothes_segmentated, clothes_mask):
    pass


def tps_transformation(clothes, tps_warp):
    pass


def shape_content_matching(clothes, clothes_segmentated, clothes_mask):
    #TODO: gets torch tensor and returns torch tensor
    tps_warp = shape_content_tps_warp(clothes_segmentated, clothes_mask)
    transformed_clothes = tps_transformation(clothes, tps_warp)
    return transformed_clothes
