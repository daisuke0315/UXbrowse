"""Local pose sensitivity to one-pixel observation noise, not physical accuracy."""
import cv2
import numpy as np


def rotation_sensitivity(xyz, rvec, tvec, matrix, distortion):
    xyz=np.asarray(xyz,np.float64)
    if len(xyz)<6:
        return None
    camera=np.einsum('ni,ji->nj',xyz,cv2.Rodrigues(np.asarray(rvec))[0])+np.asarray(tvec).reshape(3)
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):
        return None
    # Left camera-frame rotation increments avoid axis-angle singularities at
    # large world-frame rotations. Normalize translation units for the SVD.
    _,jacobian=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),matrix,distortion)
    jacobian=jacobian[:,:6].copy()
    jacobian[:,3:]*=float(np.median(camera[:,2]))
    _,s,vt=np.linalg.svd(jacobian,full_matrices=False)
    if not np.isfinite(s).all() or s[-1]<=s[0]*1e-8:
        return None
    covariance=(vt.T/(s*s))@vt
    angular_variance=float(np.linalg.eigvalsh(covariance[:3,:3])[-1])
    return float(np.degrees(np.sqrt(max(0.,angular_variance))))
