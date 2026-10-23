// Real stages, not an invented percentage. Search previews belong to the
// worker's source image, never to the latest camera frame.
const stages={queued:0,retrieval:0,landmark_bank:0,projection:0,features:1,reference:2,matching:2,landmark_matching:2,xyz:2,pnp:3,qc:3,done:3};
const labels={queued:'処理待ち',retrieval:'対象検出・DINO view選択',features:'ALIKED特徴抽出',reference:'参照特徴の読込',matching:'LightGlue照合',xyz:'対応XYZ取得',pnp:'PnP姿勢推定',qc:'品質・曖昧性チェック',done:'探索結果を現在フレームへ追跡転送',registered:'登録実写の照合・PnP'};
export function diagnosticLogSummary(log, sequence){
    if(!log)return '';
    const clock=String(log.wall_time||'').split('T')[1]?.slice(0,12)||'';
    return `記録 ${clock} · セッション ${(Number(log.elapsed_ms)/1000).toFixed(1)} s · 表示 #${sequence??'—'} · ${log.file_name}${log.error?` · ログエラー: ${log.error}`:''}${log.dropped_records?` · 記録欠落 ${log.dropped_records}件`:''}`;
}
Object.assign(labels,{landmark_bank:'実写ALIKED点索引の初回読込',projection:'現在姿勢からXYZ候補を投影',landmark_matching:'投影位置付近のALIKED点照合（view再検索なし）'});
export function searchSummary(search){
    const s=search||{},ref=s.reference_index??s.candidates?.[0]?.index;
    const active=Boolean(s.active),imageWait=!active&&s.background?.state==='image_wait';
    const waiting=imageWait||(!active&&!s.tracking&&s.background?.state==='cooldown');
    return {active,waiting,visible:active||waiting||Boolean(s.background),step:stages[s.stage]??0,
        label:imageWait?'鮮明な対象画像を待機（追跡・カメラ表示は継続）':active?(s.mode==='photo'&&s.stage==='done'?'写真の照合完了':labels[s.stage]||'ローカライズ中'):waiting?'最新映像で自動再探索します（負荷調整の待機）':'追跡継続・次の補充を待機',
        title:imageWait?'検索画像の品質待機':active?(s.mode==='photo'?'写真をローカライズ中':s.mode==='nearby'?'直前の視点付近から再ローカライズ':s.tracking?(s.mode==='guided'?'追跡継続中 · 対応点補充':'追跡継続中 · 全体を再探索'):'初期 / 再ローカライズ中'):waiting?'ロスト · 再探索待機中':'追跡・補充の状態',
        elapsed:(Math.max(0,Number(s.elapsed_ms)||0)/1000).toFixed(1),
        reference:Number.isInteger(ref)&&ref>=0?ref:null,
        referenceName:s.reference_name||s.candidates?.[0]?.image_name||'',
        detail:[active&&s.view_expansion?`参照view ${s.view_expansion.initial_top_k}→${s.view_expansion.expanded_top_k}枚で再照合`:'',
            active&&s.stage==='reference'&&s.reference_selection==='gs-real'?'同方向の実写の見た目から照合候補を選択':'',
            active&&s.stage==='reference'&&s.appearance_cache?`実写DINO索引 ${s.appearance_loaded??0}/${s.appearance_total??'—'}枚 · ${{loading:'cache読込',loaded:'cache再利用',building:'初回生成',rebuilding:'不正cacheを再生成',built:'保存済み'}[s.appearance_cache]||s.appearance_cache}`:'',
            active&&s.matching_method==='pose_landmarks'?'姿勢ベース点検索 · DINO再検索なし':'',
            active&&s.matching_method==='pose_references'?'既知姿勢の実写2枚までで領域補充 · DINO再検索なし':'',
            active&&s.stage==='landmark_bank'?`読込 ${s.bank_loaded??0}/${s.bank_total??'—'}画像（初回のみ）`:'',
            active&&s.matching_method==='pose_landmarks'&&s.projected_landmarks!=null?`投影候補 ${s.projected_landmarks}点`:'',
            active&&s.registered_reference_name?`実写 ${s.registered_reference_name}`:'',
            active&&s.retry?'XYZ付き特徴で再照合中':'',
            active&&s.candidate_total?`候補 ${s.candidate_index||0}/${s.candidate_total}`:'',
            s.reference_search&&s.matching_method==='lightglue'?`実写 ${s.reference_search.processed}/${s.reference_search.total}枚 · ${s.reference_search.early_exit?'強い姿勢一致で早期終了':s.reference_search.policy==='exhaustive'?'全候補照合':'条件不足のため候補を拡張'} · 省略 ${s.reference_search.skipped}枚`:'',
            active&&s.query_features!=null?`特徴 ${s.query_features}`:'',active&&s.geometric_inliers!=null?`幾何対応 ${s.geometric_inliers}`:'',
            active&&s.correspondences!=null?`XYZ対応 ${s.correspondences}`:'',
            active&&s.device?`AI ${s.device}`:'',
            s.seed_rejected?`前回結果の現在画像への採用を拒否: ${{stale_or_failed:'時間切れまたは照合FAIL',stale_after_transport:'追跡転送後に時間切れ',guided_prior_lost:'近傍探索中に追跡ロスト'}[s.seed_rejected]||s.seed_rejected}`:'',
            !s.tracking&&s.lost_reason?`直近の追跡終了: ${s.lost_reason}`:'',
            s.last_result?`前回 ${s.last_result.status}: ${s.last_result.detail}`:''].filter(Boolean).join(' · ')};
}

export function imageQualitySummary(search){
    const selected=search?.image_selection;if(!selected)return '';
    if(selected.waiting)return `直近${Number(selected.max_age_ms??300)} msに検索できる対象画像がありません。`;
    if(selected.age_ms==null)return '';
    const q=search.active&&search.input_quality?search.input_quality:selected.quality;
    const mode=search.photometric_mode&&search.photometric_mode!=='raw'?'軽い明度補正で再照合':'通常画像';
    return `検索入力: ${Number(selected.age_ms).toFixed(0)} ms前の画像 · ${mode} · ${selected.roi_known||search.active&&search.input_quality?'対象ROIで評価':'対象ROI未確定（最新画像のみ）'}${q?` · 鮮明度 ${Number(q.sharpness).toFixed(1)} / 明度 ${Number(q.mean).toFixed(0)} / コントラスト ${Number(q.contrast).toFixed(0)}`:''}（画像品質の目安・姿勢精度ではありません）`;
}

export function backgroundSummary(search){
    const b=search?.background;if(!b)return '';
    const states={running:search.mode==='global'?'全体探索中':'近傍補充中',cooldown:'負荷調整・待機',foreground_busy:'追跡優先で休止',ready:'検索準備',waiting:'初期位置合わせ待ち'};
    states.image_wait='鮮明な対象画像待ち';
    states.viewpoint_wait='視点変化待ち';
    const outcomes={not_started:'未実行',added:'新規点を統合',reinitialized:'追跡を再初期化',verified:'既存点の一致を再確認',search_or_transport_rejected:'探索／転送を拒否',current_frame_qc_rejected:'現在画像のQCで拒否'};
    const reasons={initial:'初期位置合わせ',viewpoint_unchanged:'同じ視点・点支持を維持',safety_review:'定期安全確認',near_search_failed:'近傍検索失敗＋追跡悪化',tracking_lost:'追跡ロスト',few_points:'点数不足',image_coverage:'画像内の分布不足',view_direction_changed:'物体を見る方向の変化',new_regions_visible:'新しく映る領域の増加',residual_rising:'再投影誤差増加', '3d_support':'3D点の広がり不足'};
    reasons.pose_jump='大きな姿勢変化';
    reasons.points_declining='点が減少・先行補充';reasons.view_direction_approaching='見る方向の変化へ先行補充';
    reasons.support_critical='点が尽きる前に優先補充';
    reasons.view_direction_emerging='新しい面へ先行補充';
    reasons.continuous_refresh='同じ視点でも継続補充';
    const reason=String(b.reason||'').split(',').map(r=>reasons[r]||r).join('・');
    const h=search.handoff;
    const handoff=h?.points!=null?` | 面の引継ぎ: 追跡${h.points}点 / 減少傾向からの予測${Number(h.forecast_points).toFixed(0)}点 · 回転${Number(h.rotation_speed_deg_s).toFixed(1)}°/s · 候補先読み${Number(h.lookahead_deg).toFixed(1)}°`:'';
    const recovery=search.recovery;
    const gap=recovery?.recovered!=null?` | 短いロストからの復帰 ${recovery.recovered}回${!search.tracking&&recovery.state==='recovering'?' · 最新画像で再追跡中':''}`:'';
    const selected={old:'旧姿勢',fresh:'新点PnP',joint:'統合PnP'};
    const correction=search.refinement?.committed?` | 新旧共通の点で姿勢更新 ${search.refinement.old_points}→${search.refinement.new_points}点 · 採用 ${selected[search.refinement.selected]||search.refinement.selected}${search.refinement.reinitialized?' · 追跡を再初期化':''}`:'';
    return `裏側更新: ${states[b.state]||b.state} · ${reason} | 前回 ${outcomes[b.outcome]||b.outcome} +${b.last_added}点 / 累計追加${b.added_total}点 · 連続拒否${b.failures} | 追跡 ${Number(b.tracking_ms).toFixed(1)} ms · 前回探索 ${Number(b.search_ms).toFixed(0)} ms${handoff}${gap}${correction}`;
}
