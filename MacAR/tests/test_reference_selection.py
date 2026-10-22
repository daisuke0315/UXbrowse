import json
from pathlib import Path
import subprocess
import unittest


class ReferenceSelectionTests(unittest.TestCase):
    def test_select_and_file_picker_share_one_path_without_mutating_other_settings(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.is_file():
            self.skipTest('Bundled Node required')
        module=Path(__file__).resolve().parents[1]/'browser_ui/reference-selection.js'
        script='''import {installReferenceSelection} from MODULE;
globalThis.Option=class {constructor(text,value){this.text=text;this.value=value;this.disabled=false;}};
const select={value:'',replaceChildren(...options){this.options=options;}},path={value:'/original',title:''};
const presets=[{id:'original',label:'original',path:'/original',available:true},
 {id:'alpha15',label:'alpha15',path:'/alpha15',available:true},
 {id:'missing',label:'missing',path:'/missing',available:false}];
const sync=installReferenceSelection(select,path,presets);
const initial=select.value;
select.value='alpha15';select.onchange();const alpha=path.value;
select.value='missing';select.onchange();const rejected=path.value;
path.value='/chosen-manifest';sync();const custom=select.value;
select.value='original';select.onchange();const restored=path.value;
console.log(JSON.stringify({initial,alpha,rejected,custom,restored,title:path.title,
 missingDisabled:select.options.find(p=>p.value==='missing').disabled}));'''.replace('MODULE',json.dumps(module.as_uri()))
        result=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script],text=True))
        self.assertEqual(result,dict(initial='original',alpha='/alpha15',rejected='/alpha15',
            custom='custom',restored='/original',title='/original',missingDisabled=True))
