const fixed=(value,digits=2)=>Number.isFinite(value)?Number(value).toFixed(digits):'—';

export function validationReasons(result){
    if(!result||result.status==='PASS')return '';
    const diagnosis=result.diagnosis||{},limits=diagnosis.limits||{};
    const explanations={
        NO_OBJECT_ROI:()=> '対象領域を検出できず、粗いview検索・局所照合へ進めませんでした',
        NO_OBJECT_FEATURES:()=> '対象領域内のALIKED特徴が不足しています',
        NO_POSE_NEAR_RC_PHOTO:()=> '粗いview候補に対応する登録実写候補がありません',
        few_inliers:()=>`採用点 ${result.inliers??'—'} 点 < 必要 ${limits.min_inliers??'—'} 点`,
        low_inlier_ratio:()=>`採用率 ${fixed((result.inlier_ratio??NaN)*100,1)}% < 必要 ${fixed((limits.min_ratio??NaN)*100,1)}%`,
        reprojection_error:()=>`平均再投影誤差 ${fixed(result.mean_reprojection_error)} px > 上限 ${fixed(limits.max_mean_px)} px`,
        reprojection_tail:()=>`誤差95パーセンタイル ${fixed(diagnosis.p95_error)} px > 上限 ${fixed(limits.max_p95_px)} px`,
        clustered_image_support:()=>`画像上の採用点が狭い範囲に集中（${fixed((diagnosis.coverage??NaN)*100,2)}% < ${fixed((limits.min_coverage??NaN)*100,2)}%）`,
        collinear_3d_support:()=>`3D点がほぼ直線状（広がり ${fixed(diagnosis.spread_3d,3)} < ${fixed(limits.min_spread_3d,3)}）`,
        behind_camera:()=> '推定姿勢で一部の3D点がカメラの後方',
        coarse_pose_mismatch:()=> `3DGS候補viewの向きとPnP姿勢が不一致${diagnosis.pose_comparison?`（${fixed(diagnosis.pose_comparison.rotation_deg,1)}° > ${fixed(diagnosis.pose_comparison.max_rotation_deg,1)}°）`:''}`,
        baseline_pose_mismatch:()=> `元PASS姿勢と面除外PnPの向きが不一致（${fixed(diagnosis.pose_comparison?.rotation_deg,1)}° > ${fixed(diagnosis.pose_comparison?.max_rotation_deg,1)}°）`,
        pose_prior_mismatch:()=> '追跡中の姿勢予測とPnP姿勢が不一致',
        ambiguous_pose:()=> '別の位置・向きにも同程度の支持があり姿勢が一意に決まらない',
        processing_error:()=>`処理エラー：${diagnosis.detail||result.detail||'詳細不明'}`,
    };
    const reasons=diagnosis.reasons||[];
    return reasons.length?reasons.map(code=>{
        const explain=explanations[code]||Object.entries(explanations).find(([key])=>code.endsWith('_'+key))?.[1];
        return explain?.()||`照合失敗：${code}`;
    }).join('／'):
        result.detail||'原因を特定できませんでした。診断ログを確認してください。';
}

export function validationStageSummary(result){
    const diagnosis=result?.diagnosis||{},stages=diagnosis.stages||{};
    const pieces=[];
    const detection=diagnosis.object_detection;
    if(detection?.trials)pieces.push(`対象検出 ${detection.trials.length}回（${detection.search_pass==='full'?'全体':detection.search_pass==='tile'?'拡大領域を採用':'拡大再検索も未検出'}）`);
    if(diagnosis.reference_matching==='all_object_features')pieces.push('実写全特徴で照合');
    if(diagnosis.reference_matching==='pose_landmarks')pieces.push('現在姿勢で実写XYZ点を投影して照合');
    if(diagnosis.xyz_only_fallback){
        const first=diagnosis.all_feature_result||{},retry=diagnosis.xyz_only_fallback;
        pieces.push(`全特徴 ${first.status??'—'}/${first.inliers??'—'}点 → XYZ付き再試行 ${retry.status}/${retry.inliers}点（${retry.adopted?'再試行を採用':'最初の結果を維持'}）`);
    }
    if(stages.reference)pieces.push(`候補 ${stages.reference}`);
    if(stages.reference_features!=null)pieces.push(`参照特徴 ${stages.reference_features}（XYZ付き ${stages.reference_xyz_features??'—'}）`);
    if(stages.raw_matches!=null)pieces.push(`局所照合 ${stages.raw_matches}`);
    if(stages.geometric_inliers!=null)pieces.push(`${diagnosis.reference_matching==='pose_landmarks'?'投影近傍・相互照合':'幾何整合'} ${stages.geometric_inliers}`);
    if(stages.xyz_correspondences!=null)pieces.push(`XYZ付与 ${stages.xyz_correspondences}`);
    if(stages.missing_xyz!=null)pieces.push(`XYZ未登録 ${stages.missing_xyz}`);
    pieces.push(`PnP採用 ${result?.inliers??'—'}`);
    const comparison=diagnosis.pose_comparison;
    if(comparison)pieces.push(`${comparison.reference==='original_pass'?'元PASS':'姿勢比較'}との向き差 ${fixed(comparison.rotation_deg,1)}°`);
    return pieces.join(' → ');
}
