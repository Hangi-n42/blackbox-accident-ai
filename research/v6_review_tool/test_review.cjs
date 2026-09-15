const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const path=require('node:path'), crypto=require('node:crypto');
const base=path.join(__dirname,'dist');
const elements={}, storage=new Map(), downloads=[], blobs=[];
function element(id){return elements[id]??=( {value:'',checked:false,textContent:'',disabled:false,children:[],listeners:{},decode:async()=>{},append(x){this.children.push(x)},addEventListener(type,fn){this.listeners[type]=fn}} );}
const html=fs.readFileSync(path.join(base,'index.html'),'utf8');
assert.ok(!/<\/option\s+[^>]/i.test(html),'Closing option tags must not carry attributes');
const radios=[];
for(const match of html.matchAll(/<input\b[^>]*>/g)){
  const attrs=Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
  if(attrs.type==='radio'){radios.push(attrs);Object.assign(element(attrs.id),{value:attrs.value,checked:/\schecked(?:\s|>)/.test(match[0])});}
}
assert.deepEqual(radios.filter(r=>r.name==='side').map(r=>r.value),['LEFT','RIGHT','']);
assert.deepEqual(radios.filter(r=>r.name==='space').map(r=>r.value),['1','0','']);
const context={Image:class{async decode(){}},window:{},document:{getElementById:element,createElement(tag){return {click(){downloads.push(this.download)},...element('new-'+tag)}},addEventListener(){}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},crypto:crypto.webcrypto,TextEncoder,Blob,URL:{createObjectURL:b=>{blobs.push(b);return 'blob:test'},revokeObjectURL(){}},structuredClone,setTimeout:fn=>fn(),console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(base,'cases.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),context);
(async()=>{
await vm.runInContext("selectCase('PUBLIC_DEV_001')",context);
assert.match(elements.scope.textContent,/독립 평가용 아님/);
assert.equal(elements.slider.max,49);
await vm.runInContext("move(49)",context);vm.runInContext("mark('contact')",context);await vm.runInContext("markAtStart()",context);
assert.match(elements.contactValue.textContent,/프레임 49/);
assert.match(elements.entryValue.textContent,/프레임 0/);
vm.runInContext("mark('contact',true)",context);
assert.equal(elements.contactValue.textContent,'판정 불가');
elements.annotator.value='TEST_ONLY'; elements.target.value='TEST_ONLY';
vm.runInContext('save()',context);
const publicCase=context.window.REVIEW_CASES.find(c=>c.ID==='PUBLIC_DEV_001');
assert.equal(JSON.parse(storage.get('v6-review-draft-v1:'+publicCase.video_sha256)).annotator,'TEST_ONLY');
await vm.runInContext('move(-999)',context);
assert.equal(elements.previous.disabled,true);
assert.equal(elements.frame.src,'cases/PUBLIC_DEV_001/frame_000000.png');
for(const asset of ['style.css','app.js','cases.js']) assert.ok(fs.existsSync(path.join(base,asset)));
if(process.argv.includes('--assets'))for(const c of context.window.REVIEW_CASES){for(const f of c.frames){const bytes=fs.readFileSync(path.join(base,f.image));assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),f.sha256);}}
function choose(name,value){const group=radios.filter(r=>r.name===name);for(const r of group)element(r.id).checked=r.value===value;element(group.find(r=>r.value===value).id).listeners.change();}
  for(const side of ['LEFT','RIGHT',''])for(const space of ['1','0','']){
    choose('side',side);choose('space',space);
    const saved=JSON.parse(storage.get('v6-review-draft-v1:'+publicCase.video_sha256));
    assert.equal(saved.side,side);assert.equal(saved.space,space);
    vm.runInContext("selectCase('PUBLIC_DEV_002');selectCase('PUBLIC_DEV_001')",context);
    for(const r of radios)assert.equal(element(r.id).checked,r.value===(r.name==='side'?side:space));
    await vm.runInContext('exportDraft()',context);
    const exported=JSON.parse(await blobs.at(-1).text());
    assert.equal(exported.review.side,side);assert.equal(exported.review.space,space);
    assert.equal(exported.record_type,'human_review_draft');assert.equal(exported.evaluation_eligible,false);
  }
  vm.runInContext("writeField('space',0);save()",context);
  assert.equal(JSON.parse(storage.get('v6-review-draft-v1:'+publicCase.video_sha256)).space,'0');
  assert.equal(downloads.length,9);
  console.log(JSON.stringify({status:'PASS',scope:'Actual HTML choice inventory plus VM persistence/restore/export; no user drafts written or browser visual QA',choice_combinations:9,checks:['all_six_visible_radio_inputs','development_scope','frame_boundary','contact_mark','entry_at_start','uncertain','save_restore_all_choices','export_all_choices','numeric_zero_preserved'],asset_hashes_checked:process.argv.includes('--assets')}));
})().catch(error=>{console.error(error);process.exitCode=1;});
