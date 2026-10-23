"""Ceres BA with fixed CT points, and separate held-out evaluation."""
import numpy as np
import pycolmap as pc

from .geometry import similarity, triangulate


def statistics(values, failures=0):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return dict(count=len(finite), failures=int(failures + len(values)-len(finite)),
                mean=float(np.mean(finite)) if len(finite) else None,
                std=float(np.std(finite)) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                maximum=float(np.max(finite)) if len(finite) else None)


def observations(model, point):
    by_name = {im.name: im for im in model.images.values() if im.has_pose}
    result, seen = [], set()
    for observation in point.get('observations', []):
        name = observation['image_id']
        uv = np.asarray(observation['uv'], float)
        if name in seen or uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError('Invalid or duplicate GCP image observation')
        seen.add(name)
        if name in by_name:
            image = by_name[name]
            camera = model.cameras[image.camera_id]
            if not (0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height):
                raise ValueError('GCP UV outside image')
            result.append((image, camera, uv))
    return result


def triangulate_point(model, point):
    obs = observations(model, point)
    poses = [im.cam_from_world().matrix() for im, _, _ in obs]
    uv = [cam.cam_from_img(pixel) for _, cam, pixel in obs]
    if any(item is None for item in uv):
        raise ValueError('Cannot unproject GCP observation')
    return triangulate(poses, uv)


def align_to_ct(model, points):
    controls = [p for p in points if p.get('enabled', True) and p['role'] == 'control']
    source = [triangulate_point(model, p) for p in controls]
    target = np.array([p['xyz'] for p in controls], float)
    scale, rotation, translation = similarity(source, target)
    aligned = pc.Reconstruction(model)
    aligned.transform(pc.Sim3d(scale, pc.Rotation3d(rotation), translation))
    errors = np.linalg.norm(scale * np.asarray(source) @ rotation.T + translation - target, axis=1)
    return aligned, dict(direction='CT_LPS_from_SfM', scale=scale, rotation=rotation.tolist(),
                         translation_mm=translation.tolist(), residual_mm=statistics(errors),
                         gcp_ids=[p['id'] for p in controls])


def projection_error(image, camera, xyz, uv):
    local = image.cam_from_world() * np.asarray(xyz, float)
    if local[2] <= 0:
        return float('nan')
    pixel = camera.img_from_cam(local)
    return float(np.linalg.norm(pixel - uv)) if pixel is not None else float('nan')


def reprojection_errors(model):
    errors = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        camera = model.cameras[image.camera_id]
        for point in image.points2D:
            if point.has_point3D():
                errors.append(projection_error(image, camera, model.points3D[point.point3D_id].xyz, point.xy))
    return statistics(errors)


def constrained_bundle(model, points, refine_intrinsics=False, max_iterations=200, gcp_max_error_px=4.):
    controls = [p for p in points if p.get('enabled', True) and p['role'] == 'control']
    if len({p['id'] for p in controls}) != len(controls):
        raise ValueError('Duplicate GCP IDs')
    # Validate geometry before allowing fixed GCPs to resolve the BA gauge.
    xyz = np.asarray([p['xyz'] for p in controls], float)
    similarity(xyz, xyz)
    for point in controls:
        if len(observations(model, point)) < 2:
            raise ValueError('Every GCP needs observations in two registered images')
    result = pc.Reconstruction(model)
    config = pc.BundleAdjustmentConfig()
    for image_id in result.reg_image_ids():
        config.add_image(image_id)
    fixed = {}
    for point in controls:
        track = pc.Track()
        for image, _, uv in observations(result, point):
            index = image.num_points2D()
            image.points2D.append(pc.Point2D(uv))
            track.add_element(image.image_id, index)
        point_id = result.add_point3D(np.asarray(point['xyz'], float), track, np.array([255, 180, 0], np.uint8))
        config.add_constant_point(point_id)
        fixed[point['id']] = int(point_id)
    # UNSPECIFIED does not add an arbitrary camera/point gauge; fixed GCPs supply it.
    options = pc.BundleAdjustmentOptions()
    options.refine_focal_length = refine_intrinsics
    options.refine_principal_point = refine_intrinsics
    options.refine_extra_params = refine_intrinsics
    options.refine_sensor_from_rig = False
    options.min_track_length = 0
    options.print_summary = False
    options.ceres.loss_function_type = pc.LossFunctionType.HUBER
    options.ceres.loss_function_scale = 2.
    options.ceres.solver_options.max_num_iterations = max_iterations
    options.ceres.solver_options.function_tolerance = 1e-10
    options.ceres.solver_options.gradient_tolerance = 1e-8
    before = reprojection_errors(result)
    summary = pc.create_default_bundle_adjuster(options, config, result).solve()
    converged = (summary.termination_type == pc.BundleAdjustmentTerminationType.CONVERGENCE
                 and summary.is_solution_usable())
    for point in controls:
        if not np.array_equal(result.points3D[fixed[point['id']]].xyz, point['xyz']):
            raise RuntimeError('Fixed CT GCP moved during optimization')
    gcp_errors = [projection_error(im, cam, p['xyz'], uv)
                  for p in controls for im, cam, uv in observations(result, p)]
    deltas = []
    for image_id in result.reg_image_ids():
        previous, current = model.images[image_id].cam_from_world(), result.images[image_id].cam_from_world()
        angle = np.degrees(np.arccos(np.clip((np.trace(current.rotation.matrix() @ previous.rotation.matrix().T)-1)/2, -1, 1)))
        deltas.append(dict(image_id=int(image_id), rotation_deg=float(angle),
                           center_shift_mm=float(np.linalg.norm(result.images[image_id].projection_center()-model.images[image_id].projection_center()))))
    result.update_point_3d_errors()
    return result, dict(converged=bool(converged), solver=summary.brief_report(),
                        termination=str(summary.termination_type), fixed_gcp_ids=fixed,
                        before_reprojection_px=before, after_reprojection_px=reprojection_errors(result),
                        gcp_reprojection_px=statistics(gcp_errors), camera_changes=deltas,
                        gcp_outlier_observations=int(np.sum(np.asarray(gcp_errors) > gcp_max_error_px)),
                        gcp_max_error_px=gcp_max_error_px,
                        warnings=['Fewer than four control GCPs'] if len(controls) < 4 else [])


def evaluate_points(model, points):
    rows, tre, reprojection, original_tre, interpolated_tre = [], [], [], [], []
    enabled = [p for p in points if p.get('enabled', True)]
    interpolated_controls = [p['id'] for p in enabled if p['role']=='control' and
                             p.get('ct_sampling', {}).get('status')=='interpolated']
    reference_uncertain = any(p.get('ct_sampling', {}).get('status') != 'original' for p in enabled)
    for point in points:
        if not point.get('enabled', True) or point['role'] != 'evaluation':
            continue
        try:
            xyz = triangulate_point(model, point)
            error = float(np.linalg.norm(xyz-np.asarray(point['xyz'])))
            pixel_errors = [projection_error(im, cam, point['xyz'], uv)
                            for im, cam, uv in observations(model, point)]
            rows.append(dict(id=point['id'], reconstructed_xyz=xyz.tolist(), ct_xyz=point['xyz'], tre_mm=error,
                             ct_reprojection_px=statistics(pixel_errors), status='measured',
                             ct_sampling=point.get('ct_sampling', {'status':'unavailable'})))
            tre.append(error)
            if point.get('ct_sampling', {}).get('status')=='original':
                original_tre.append(error)
            elif point.get('ct_sampling', {}).get('status')=='interpolated':
                interpolated_tre.append(error)
            reprojection.extend(pixel_errors)
        except ValueError as error:
            rows.append(dict(id=point['id'], status='failed', reason=str(error),
                             ct_sampling=point.get('ct_sampling', {'status':'unavailable'})))
    failed = sum(row['status'] == 'failed' for row in rows)
    measured = statistics(tre, failed)
    status = 'unverified'
    if tre:
        status = 'within_provisional_target' if not failed and measured['mean'] <= 2 and measured['maximum'] <= 3 else 'failed'
        if reference_uncertain:
            status = 'unverified'
    return dict(accuracy_status=status, evaluation_count=len(rows), tre_mm=measured,
                evaluation_reprojection_px=statistics(reprojection), points=rows,
                original_reference_tre_mm=statistics(original_tre),
                interpolated_reference_tre_mm=statistics(interpolated_tre),
                interpolated_control_ids=interpolated_controls,
                reference_sampling_uncertain=reference_uncertain,
                performance_target_validated=False)
