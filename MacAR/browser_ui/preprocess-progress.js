// Presentation only. No thresholds, pose, XYZ or job scheduling changes.
export const phases=['load','matching','triangulation','export','audit'];
const phaseLabels={launch:'起動中',load:'特徴・カメラを読み込み',inspect:'元地図を確認',models:'照合モデルを準備',matching:'画像同士を照合',triangulation:'対応点のXYZを確定',export:'地図ファイルを保存',audit:'全XYZの幾何監査',pass_complete:'監査合格・保存完了'};
const units={matching:'画像ペア',triangulation:'特徴点トラック',export:'登録画像',audit:'登録画像',inspect:'登録画像'};
const number=n=>Number(n||0).toLocaleString('ja-JP');
export function shortMap(path){return path?String(path).split('/').filter(Boolean).slice(-2).join('/'):'未指定';}
export function jobProgress(job){
    const saved=job.passes||[],finished=['complete','no_changes'].includes(job.status);
    // Older workers report enrichment complete while the outer audit is still
    // pending. Do not display that as a finished job or reuse export counters.
    const legacyComplete=job.stage==='complete'&&!finished;
    const phase=legacyComplete?'audit':job.stage;
    const total=legacyComplete?0:Number(job.total||0),done=legacyComplete?0:Number(job.completed||0);
    const unit=units[phase],fraction=unit&&total>0?Math.max(0,Math.min(1,done/total)):null;
    const max=Number(job.config?.max_neighbors||job.neighbors||48);
    const schedule=[...new Set([12,Math.min(24,max),max])].sort((a,b)=>a-b);
    const active=Number(job.neighbors||schedule[Math.max(0,(job.pass_number||1)-1)]);
    const stages=schedule.map(neighbors=>{
        const snapshot=saved.find(s=>s.neighbors===neighbors);
        const status=snapshot?'saved':finished?'skipped':neighbors===active&&job.pass_number?'current':'pending';
        return {neighbors,status,snapshot};
    });
    const process=finished?'前処理完了':phaseLabels[phase]||'準備中';
    const status={running:'実行中',paused:'一時停止中',interrupted:'停止済み（未完了）',cancelled:'中止済み',error:'エラー',complete:'完了・監査合格',no_changes:'追加処理不要'}[job.status]||job.status;
    const counts=job.counts||{};
    const cached=Number(counts.cached_pairs||0),computed=Number(counts.computed_pairs||0);
    const seconds=Math.max(0,Math.floor(Number(job.elapsed_seconds||0)));
    const duration=seconds<60?`${seconds}秒`:`${Math.floor(seconds/60)}分${seconds%60}秒`;
    return {status,phase,process,fraction,stages,finished,duration,
        phaseIndex:phase==='pass_complete'?phases.length:phases.indexOf(phase==='models'?'load':phase),
        counter: fraction===null?'処理中（件数未確定）':`${number(done)} / ${number(total)} ${unit} · ${(fraction*100).toFixed(1)}%`,
        cache:cached+computed?`照合済みキャッシュを再利用 ${number(cached)}ペア ／ 新規計算 ${number(computed)}ペア`:null,
        savedText:`監査合格・保存済み ${saved.length} / ${schedule.length} 段階`,
        latest:saved.at(-1)||null,source:shortMap(job.source||job.config?.source),
        warning: job.status==='error'?job.detail:null,
    };
}
