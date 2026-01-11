from light_glue.lightglue.lightglue import LightGlue
from light_glue.lightglue.disk import DISK
from light_glue.lightglue.utils import load_image, rbd

def light_glue_checker(extractor, matcher, device, image_path0, image_path1, score_threshold=0.7):
    image0 = load_image(image_path0).to(device)
    image1 = load_image(image_path1).to(device)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # --- match ---
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # =========================
    # SCORE THRESHOLDING (0.7)
    # =========================
    score_thresh = 0.8

    scores  = matches01['scores']     # (K,)
    matches = matches01['matches']    # (K, 2)

    mask = scores > score_thresh

    matches = matches[mask]
    scores  = scores[mask]

    # update matches01 so viz2d uses filtered matches
    matches01['matches'] = matches
    matches01['scores']  = scores

    return matches01