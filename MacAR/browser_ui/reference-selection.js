// One setting: presets and the native manifest picker both update views_path.
export function installReferenceSelection(select, pathInput, presets) {
    const options=presets.map(preset=>{
        const option=new Option(preset.label+(preset.available?'':' · 未生成'),preset.id);
        option.disabled=!preset.available;
        return option;
    });
    const custom=new Option('ファイル指定（詳細設定）','custom');
    custom.disabled=true;
    select.replaceChildren(...options,custom);
    const sync=()=>{
        select.value=presets.find(preset=>preset.path===pathInput.value)?.id||'custom';
        pathInput.title=pathInput.value;
    };
    select.onchange=()=>{
        const preset=presets.find(preset=>preset.id===select.value);
        if(preset?.available)pathInput.value=preset.path;
        sync();
    };
    sync();
    return sync;
}
