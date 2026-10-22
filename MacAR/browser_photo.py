"""Still-photo decoding and single-shot matching helpers for the browser UI."""
import io

import cv2
import numpy as np
from PIL import Image

MAX_PHOTO_BYTES=64*1024*1024
MAX_PHOTO_PIXELS=60_000_000
PHOTO_RETRIEVAL_MAX_WIDTH=1280


def decode_photo(data):
    if not data or len(data)>MAX_PHOTO_BYTES:
        raise ValueError('写真は64 MB以下のJPEG・PNGを選んでください。')
    try:
        with Image.open(io.BytesIO(data)) as header:
            width,height=header.size
            # iPhone JPEGs may contain auxiliary HDR/gain-map images and be
            # identified as MPO. OpenCV decodes their primary JPEG frame.
            if header.format not in ('JPEG','MPO','PNG') or width*height>MAX_PHOTO_PIXELS or max(width,height)>16000:
                raise ValueError('JPEG・PNG（6000万画素以下、1辺16000 px以下）を選んでください。')
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('写真を読み込めません。JPEG・PNGを選んでください。') from exc
    # IMREAD_COLOR applies JPEG EXIF orientation. K is then built for this
    # upright image, and all subsequent observations use its pixel coordinates.
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('写真のデコードに失敗しました。')
    return image


def photo_engine(wrapper, feature_size):
    """Configure appearance analysis only; localization still receives native UV/K."""
    engine=getattr(wrapper,'global_localizer',None) or getattr(wrapper,'localizer',wrapper)
    engine.query_resize=int(feature_size)
    engine.retriever.query_max_width=PHOTO_RETRIEVAL_MAX_WIDTH
    return engine


def localize_photo(engine, image, camera_matrix, distortion, max_views=10):
    """Release unused MPS memory, then match with a photo-only retry budget."""
    if getattr(getattr(engine,'device',None),'type',None)=='mps':
        engine.torch.mps.synchronize()
        engine.torch.mps.empty_cache()
    initial_top_k=getattr(engine,'top_k',None)
    result,canvas=engine.localize(image,camera_matrix,distortion)
    engine.photo_view_expansion=None
    if (result.status=='PASS' or getattr(engine,'source_type',None)!='3dgs'
            or initial_top_k is None):
        return result,canvas
    expanded_top_k=min(max_views,len(getattr(engine,'views',())))
    if expanded_top_k<=initial_top_k:
        return result,canvas
    engine.photo_view_expansion=dict(initial_top_k=initial_top_k,
                                     expanded_top_k=expanded_top_k,
                                     initial_inliers=result.inliers)
    print(f'Photo: {initial_top_k} reference views yielded {result.inliers} inliers; '
          f'retrying with {expanded_top_k} views',flush=True)
    engine.top_k=expanded_top_k
    try:
        return engine.localize(image,camera_matrix,distortion)
    finally:
        engine.top_k=initial_top_k
