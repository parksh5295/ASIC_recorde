import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)

def maybe_downcast_float32(array, enabled, label):
    """
    Optionally convert a floating-point array (NumPy array or pandas DataFrame) to float32.
    """
    if not enabled or array is None:
        return array

    if isinstance(array, pd.DataFrame):
        float_cols = array.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            array[float_cols] = array[float_cols].astype(np.float32)
            logger.info(f"[Float32] DataFrame '{label}'의 {len(float_cols)}개 부동소수 컬럼을 float32로 변환했습니다.")
        return array

    if hasattr(array, "dtype") and np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float32)
        logger.info(f"[Float32] '{label}' 배열을 float32로 다운캐스팅했습니다. (shape={array.shape})")
    return array
