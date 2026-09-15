'use strict';
const cases = window.REVIEW_CASES || [];
const $ = id => document.getElementById(id);
let active = null, index = 0, draft = {}, loadRevision = 0, displayed = null;
const fields = ['annotator','target','side','space','notes'];
const choiceIds = {side:['side-left','side-right','side-unknown'],space:['space-yes','space-no','space-unknown']};
function readField(name){return choiceIds[name]?choiceIds[name].map($).find(input=>input.checked)?.value??'':$(name).value;}
function writeField(name,value){value=String(value??'');if(choiceIds[name]){const inputs=choiceIds[name].map($);if(!inputs.some(input=>input.value===value))value='';for(const input of inputs)input.checked=input.value===value;}else $(name).value=value;}
function key(){return 'v6-review-draft-v1:' + active.video_sha256;}
function save(){if(!active)return;for(const f of fields)draft[f]=readField(f);draft.annotation_blinded=$('blind').checked;try{localStorage.setItem(key(),JSON.stringify(draft));}catch(e){$('status').textContent='브라우저 저장을 사용할 수 없습니다. 파일로 저장하세요.';}}
async function show(){
  if(!active)return false;
  const revision=++loadRevision, selectedCase=active, selectedIndex=index, f=selectedCase.frames[selectedIndex];
  displayed=null;$('frame').hidden=true;$('contact').disabled=true;$('entry').disabled=true;
  $('slider').value=index;$('position').textContent=`원본 프레임 ${f.frame} 불러오는 중 · 기록은 표시 완료 후 가능합니다`;
  $('previous').disabled=index===0;$('next').disabled=index===active.frames.length-1;
  try{
    // Separate requests prevent a late previous load from certifying the new selection.
    const pending=new Image();pending.src=f.image;await pending.decode();
    if(revision!==loadRevision||active!==selectedCase||index!==selectedIndex)return false;
    $('frame').src=f.image;await $('frame').decode();
    if(revision!==loadRevision||active!==selectedCase||index!==selectedIndex)return false;
    displayed={selectedCase,index:selectedIndex,frame:f,revision};$('frame').hidden=false;
    $('position').textContent=`원본 프레임 ${f.frame} / ${active.frames.length-1} · 실제 표시 시각 ${f.pts_seconds.toFixed(6)}초`;
    $('contact').disabled=false;$('entry').disabled=false;return true;
  }catch(error){if(revision===loadRevision){$('position').textContent=`원본 프레임 ${f.frame} 표시 실패 · 다시 이동하여 불러오세요`;$('status').textContent='이미지를 표시하지 못해 현재 프레임을 기록할 수 없습니다.';}return false;}
}
function mark(name,unknown=false,start=false){
  if(!active)return false;
  if(!unknown&&(!displayed||displayed.selectedCase!==active||displayed.index!==index||displayed.revision!==loadRevision||(start&&index!==0))){$('status').textContent='선택한 프레임의 표시가 완료된 뒤 기록하세요.';return false;}
  const f=displayed?.frame;
  draft[name]=unknown?{status:'uncertain',frame:null,pts_seconds:null}:{status:'observed',frame:f.frame,pts_seconds:f.pts_seconds};if(name==='entry')draft.already_entered_at_start=start;save();renderLabels();return true;
}
async function markAtStart(){if(!active)return;index=0;if(await show())mark('entry',false,true);}
function renderLabels(){for(const [name,label] of [['contact','contactValue'],['entry','entryValue']]){const v=draft[name];$(label).textContent=!v?'미기록':v.status==='uncertain'?'판정 불가':`프레임 ${v.frame} · ${v.pts_seconds.toFixed(6)}초`+(name==='entry'&&draft.already_entered_at_start?' · 시작부터 진입':'');}}
function selectCase(value){save();active=cases.find(c=>c.ID===value);index=0;draft={};try{draft=JSON.parse(localStorage.getItem(key())||'{}');}catch(e){draft={};}for(const f of fields)writeField(f,draft[f]);$('blind').checked=draft.annotation_blinded===true;$('slider').max=active.frames.length-1;$('scope').textContent=active.exposure==='seen'?'이미 사용한 개발 영상 · 독립 평가용 아님':'미배정 영상 · 원천 검증 필요';renderLabels();return show();}
function move(delta){index=Math.max(0,Math.min(active.frames.length-1,index+delta));return show();}
async function sha(text){const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text));return Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('');}
async function exportDraft(){save();if(!draft.annotator?.trim()||!draft.target?.trim()){ $('status').textContent='검수자와 실제 접촉 상대 설명을 입력하세요. 상대를 식별할 수 없으면 그 사유를 적어 주세요.';return;}
const snapshot={case:structuredClone(active),review:structuredClone(draft)};
const c=snapshot.case;
const result={schema_version:1,record_type:'human_review_draft',evaluation_eligible:false,created_at:new Date().toISOString(),ID:c.ID,source_group_id:c.source_group_id,source_video_sha256:c.video_sha256,source_uri:c.source_uri,split:c.split,exposure:c.exposure,time_origin:c.time_origin,frame_mapping_sha256:await sha(JSON.stringify(c.frames)),review:snapshot.review,requires:'Independent review and documented adjudication; no automatic promotion to ground truth.'};
const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=c.ID+'_review_'+Date.now()+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').textContent='검수 초안 파일을 저장했습니다. 확정 정답으로 승격하지 않았습니다.';}
for(const c of cases){const option=document.createElement('option');option.value=c.ID;option.textContent=c.ID;$('case').append(option);}
$('case').addEventListener('change',e=>selectCase(e.target.value));$('slider').addEventListener('input',e=>{index=Number(e.target.value);show();});$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('contact').onclick=()=>mark('contact');$('contactUnknown').onclick=()=>mark('contact',true);$('entry').onclick=()=>mark('entry');$('atStart').onclick=()=>markAtStart();$('entryUnknown').onclick=()=>mark('entry',true);$('export').onclick=()=>exportDraft().catch(e=>{$('status').textContent='저장 실패: '+e.message;});for(const f of [...fields.filter(f=>!choiceIds[f]),'blind'])$(f).addEventListener('input',save);for(const ids of Object.values(choiceIds))for(const id of ids)$(id).addEventListener('change',save);
document.addEventListener('keydown',e=>{if(!active||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();move((e.key==='ArrowLeft'?-1:1)*(e.shiftKey?10:1));}});
if(cases.length)selectCase(cases[0].ID);else $('status').textContent='등록된 영상이 없습니다. 원본과 시각 대응 자료를 먼저 준비하세요.';
if(document.modelContext?.registerTool){Promise.resolve(document.modelContext.registerTool({name:'read_review_progress',description:'Read current frame and whether a draft exists. Does not annotate or certify human ground truth.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:true},execute:()=>({case:active?.ID,frame:active?.frames[index]?.frame,record_type:'human_review_draft',evaluation_eligible:false,contact_recorded:!!draft.contact,entry_recorded:!!draft.entry})})).catch(()=>{});}

