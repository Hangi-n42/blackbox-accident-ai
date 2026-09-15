'use strict';
// TEST_ONLY synthetic UI. No actual cases, local browser drafts, or review JSON writes.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path'),crypto=require('node:crypto');
const pending=[],visible=[],elements={},storage=new Map(),blobs=[],downloads=[];
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return{promise,resolve,reject};}
function element(id){return elements[id]??={value:'',checked:false,textContent:'',disabled:false,hidden:false,children:[],append(x){this.children.push(x)},addEventListener(){},decode(){const d=deferred();visible.push({src:this.src,...d});return d.promise;}};}
const ids=['case','frame','slider','position','previous','next','contact','entry','atStart','contactUnknown','entryUnknown','export','blind','scope','status','contactValue','entryValue','annotator','target','notes'];for(const id of ids)element(id);
for(const [kind,vals] of Object.entries({side:['LEFT','RIGHT',''],space:['1','0','']}))for(let i=0;i<3;i++){const id=kind+'-'+(kind==='side'?['left','right','unknown']:['yes','no','unknown'])[i];Object.assign(element(id),{value:vals[i],checked:i===2});}
function fixture(ID){return{ID,video_sha256:ID,source_group_id:'TEST_ONLY',source_uri:'TEST_ONLY',frames:[0,1,2].map(frame=>({frame,pts_seconds:frame/10,image:ID+'-'+frame+'.png',sha256:'TEST_ONLY'}))};}
const ctx={window:{REVIEW_CASES:[fixture('TEST_A'),fixture('TEST_B')]},Image:class{decode(){const d=deferred();pending.push({src:this.src,...d});return d.promise;}},document:{getElementById:element,createElement(){return{click(){downloads.push(this.download)}}},addEventListener(){}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},crypto:crypto.webcrypto,TextEncoder,Blob,structuredClone,URL:{createObjectURL:b=>{blobs.push(b);return'test'},revokeObjectURL(){}},setTimeout:fn=>fn()};
vm.createContext(ctx);const run=s=>vm.runInContext(s,ctx);vm.runInContext(fs.readFileSync(path.join(__dirname,'dist/app.js'),'utf8'),ctx);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  assert.equal(elements.frame.hidden,true);assert.equal(elements.contact.disabled,true);assert.equal(run("mark('contact')"),false);
  pending.shift().resolve();await tick();assert.equal(elements.contact.disabled,true,'preload completion alone must not permit marking');
  visible.shift().resolve();await tick();assert.equal(elements.frame.hidden,false);assert.equal(run("mark('contact')"),true);assert.equal(run('draft.contact.frame'),0);
  run('move(1)');const first=pending.shift();run('move(1)');const latest=pending.shift();
  assert.equal(run("mark('entry')"),false);assert.equal(elements.frame.hidden,true);
  latest.resolve();await tick();visible.shift().resolve();await tick();assert.equal(run("mark('entry')"),true);assert.equal(run('draft.entry.frame'),2);
  first.resolve();await tick();assert.equal(visible.length,0,'stale preload must not reach displayed image');assert.match(elements.position.textContent,/프레임 2/);assert.equal(elements.frame.src,'TEST_A-2.png');
  run('move(-1)');pending.shift().resolve();await tick();const staleVisible=visible.shift();
  run("selectCase('TEST_B')");pending.shift().resolve();await tick();visible.shift().resolve();await tick();staleVisible.resolve();await tick();
  assert.equal(run("mark('contact')"),true);assert.equal(run('active.ID'),'TEST_B');assert.equal(run('draft.contact.frame'),0);assert.equal(elements.frame.src,'TEST_B-0.png');
  run('move(1)');pending.shift().reject(new Error('TEST_ONLY load failure'));await tick();assert.equal(elements.contact.disabled,true);assert.equal(run("mark('contact')"),false);assert.equal(run('draft.contact.frame'),0);
  assert.equal(run("mark('entry',true)"),true,'uncertain recording does not assert any displayed frame');assert.equal(run('draft.entry.frame'),null);
  const start=run('markAtStart()');pending.shift().resolve();await tick();visible.shift().resolve();await start;assert.equal(run('draft.entry.frame'),0);assert.equal(run('draft.already_entered_at_start'),true);
  elements.annotator.value='TEST_ONLY';elements.target.value='TEST_B_TARGET';const exporting=run('exportDraft()');
  run("selectCase('TEST_A')");elements.target.value='TEST_A_CHANGED';run('save()');await exporting;
  const result=JSON.parse(await blobs.at(-1).text());assert.equal(result.ID,'TEST_B');assert.equal(result.review.target,'TEST_B_TARGET');assert.match(downloads.at(-1),/^TEST_B_review_/);
  assert.equal(result.frame_mapping_sha256,crypto.createHash('sha256').update(JSON.stringify(ctx.window.REVIEW_CASES[1].frames)).digest('hex'));
  console.log(JSON.stringify({status:'PASS',scope:'TEST_ONLY controlled decode promises; not real browser visual QA',checks:['slow_preload_and_visible_decode','rapid_navigation_stale_preload','stale_visible_decode_across_cases','load_failure_blocks_mark','uncertain_preserved','at_start_waits_for_frame_zero','export_atomic_snapshot']}));
})().catch(e=>{console.error(e);process.exitCode=1;});
