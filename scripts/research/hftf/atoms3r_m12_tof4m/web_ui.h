#pragma once

#include <Arduino.h>

constexpr char kDashboardHtml[] PROGMEM = R"HTML(
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AtomS3R 实时测距</title>
  <style>
    :root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
    *{box-sizing:border-box}body{margin:0;background:#07111f;color:#f4f7fb}
    main{width:min(100%,1100px);margin:auto;padding:16px}.top,.actions,.foot{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
    h1{font-size:clamp(20px,4vw,30px);margin:4px 0}.badges{display:flex;gap:8px;flex-wrap:wrap}.pill{padding:7px 11px;border-radius:999px;background:#17304c;color:#bfe4ff;font-size:14px}
    .viewer{position:relative;margin-top:14px;background:#000;border-radius:18px;overflow:hidden;aspect-ratio:4/3;box-shadow:0 18px 60px #0008}
    #stream{width:100%;height:100%;object-fit:contain;display:block}.reticle{position:absolute;left:39%;top:35%;width:22%;height:30%;border:3px solid #43f5a3;border-radius:14px;box-shadow:0 0 0 9999px #0002,0 0 18px #43f5a399;pointer-events:none}
    .reading{position:absolute;left:50%;bottom:7%;transform:translateX(-50%);min-width:180px;text-align:center;background:#06121ddd;border:1px solid #ffffff38;border-radius:14px;padding:10px 18px;backdrop-filter:blur(8px)}
    #distance{font-size:clamp(30px,7vw,54px);font-weight:750;line-height:1.05}.label{font-size:13px;color:#b8c7d8;margin-top:5px}.invalid #distance{color:#ffc15a}.valid #distance{color:#63ffb4}
    .notice{display:none;margin-top:12px;padding:11px 14px;border-radius:12px;background:#50291c;color:#ffd7bd}.notice.show{display:block}
    .panel{margin-top:16px;padding:16px;background:#0d1b2b;border:1px solid #ffffff18;border-radius:16px}.panel h2{font-size:19px;margin:0 0 14px}
    .controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:13px}.field label{display:block;color:#aebed0;font-size:13px;margin-bottom:6px}.field select,.field input{width:100%;height:40px;border:1px solid #ffffff2b;border-radius:9px;background:#071421;color:#fff;padding:7px}.check{display:flex;gap:9px;align-items:center;min-height:40px}.check input{width:20px;height:20px}
    button,.linkbtn{border:1px solid #ffffff33;border-radius:10px;background:#17283b;color:#eef6ff;padding:10px 14px;cursor:pointer;text-decoration:none;font:inherit}.primary{background:#1fb97a;color:#031e14;border-color:#3ee7a2;font-weight:700}button:disabled{opacity:.55;cursor:wait}
    #actionState{color:#aebed0;font-size:14px}.foot{margin-top:14px;color:#9eb0c4;font-size:14px}.foot p{margin:0;max-width:760px}
    @media(max-width:600px){main{padding:10px}.panel{padding:13px}.reading{bottom:4%}}
  </style>
</head>
<body><main>
  <div class="top"><h1>AtomS3R 实时画面＋距离</h1><div class="badges"><span class="pill" id="cameraBadge">读取相机...</span><span class="pill" id="fpsBadge">-- fps</span><span class="pill" id="state">正在连接测距...</span></div></div>
  <div class="viewer"><img id="stream" alt="AtomS3R-M12 实时摄像头画面">
    <div class="reticle" aria-hidden="true"></div>
    <div class="reading invalid" id="reading"><div id="distance">--</div><div class="label">中央单点 ToF 距离</div></div>
  </div>
  <div class="notice" id="notice"></div>
  <section class="panel"><h2>相机控制</h2>
    <div class="controls">
      <div class="field"><label for="resolution">分辨率</label><select id="resolution"><option>VGA</option><option>SVGA</option><option selected>XGA</option><option>SXGA</option><option>UXGA</option></select></div>
      <div class="field"><label for="quality">JPEG 质量（数值越小越清晰） <span id="qualityValue">10</span></label><input id="quality" type="range" min="6" max="30" value="10"></div>
      <div class="field"><label for="brightness">亮度</label><select id="brightness"><option value="-2">-2</option><option value="-1">-1</option><option value="0">0</option><option value="1" selected>+1</option><option value="2">+2</option></select></div>
      <div class="field"><label>曝光模式</label><div class="check"><input id="autoExposure" type="checkbox" checked><label for="autoExposure">自动曝光</label></div></div>
      <div class="field"><label for="compensation">曝光补偿</label><select id="compensation"><option value="-2">-2</option><option value="-1">-1</option><option value="0" selected>0</option><option value="1">+1</option><option value="2">+2</option></select></div>
      <div class="field"><label for="manualExposure">手动曝光值（0–1200）</label><input id="manualExposure" type="number" min="0" max="1200" value="300"></div>
    </div>
    <div class="actions" style="margin-top:14px"><div><button class="primary" id="applyCamera">应用参数</button> <button id="capture">下载截图＋JSON</button> <a class="linkbtn" href="/status">设备状态</a></div><span id="actionState">参数仅在本次开机期间生效，重启恢复 XGA 默认值。</span></div>
  </section>
  <div class="foot"><p>绿色框仅表示 ToF4M 的中央窄视场；数值不代表整幅图的深度，也不能作为独立安全判断。</p>
    <form method="post" action="/forget" onsubmit="return confirm('清除 Wi-Fi 配置并重新进入设置模式？')"><button type="submit">更换 Wi-Fi</button></form>
  </div>
</main><script>
  const $=id=>document.getElementById(id);
  const img=$('stream'),reading=$('reading'),distance=$('distance'),state=$('state'),notice=$('notice'),actionState=$('actionState');
  const resolution=$('resolution'),quality=$('quality'),brightness=$('brightness'),autoExposure=$('autoExposure'),compensation=$('compensation'),manualExposure=$('manualExposure');
  const staticMode=new URLSearchParams(location.search).has('static');
  let streamRetryMs=1000,streamTimer=0,rangeFailures=0,statusFailures=0,lastHealthyStream=Date.now();
  function showNotice(message){notice.textContent=message;notice.className=message?'notice show':'notice'}
  function connectStream(reason){
    clearTimeout(streamTimer);
    if(reason){showNotice(reason)}
    if(staticMode){img.src='/api/snapshot?t='+Date.now();return}
    img.src=''; img.src='http://'+location.hostname+':81/stream?t='+Date.now();
  }
  img.onload=()=>{streamRetryMs=1000;lastHealthyStream=Date.now();showNotice('')};
  img.onerror=()=>{
    if(staticMode){showNotice('静态诊断画面加载失败。');return}
    showNotice('实时画面断开，'+Math.ceil(streamRetryMs/1000)+' 秒后自动重连。');
    clearTimeout(streamTimer);streamTimer=setTimeout(()=>connectStream('正在重新连接实时画面...'),streamRetryMs);
    streamRetryMs=Math.min(streamRetryMs*2,10000);
  };
  quality.oninput=()=>{$('qualityValue').textContent=quality.value};
  function syncExposureUi(){compensation.disabled=!autoExposure.checked;manualExposure.disabled=autoExposure.checked}
  autoExposure.onchange=syncExposureUi;syncExposureUi();
  async function loadCamera(){
    try{
      const c=await (await fetch('/api/camera',{cache:'no-store'})).json();
      resolution.value=c.resolution;quality.value=c.jpeg_quality;$('qualityValue').textContent=c.jpeg_quality;
      brightness.value=String(c.brightness);autoExposure.checked=c.auto_exposure;compensation.value=String(c.exposure_compensation);manualExposure.value=c.manual_exposure;syncExposureUi();
      $('cameraBadge').textContent=c.resolution+' · '+c.width+'×'+c.height;
    }catch(e){actionState.textContent='相机参数读取失败，正在保留默认值。'}
  }
  $('applyCamera').onclick=async()=>{
    const button=$('applyCamera');button.disabled=true;actionState.textContent='正在应用参数...';
    const params=new URLSearchParams({resolution:resolution.value,quality:quality.value,brightness:brightness.value,auto_exposure:autoExposure.checked?'1':'0',exposure_compensation:compensation.value,manual_exposure:manualExposure.value});
    try{
      const response=await fetch('/api/camera',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:params});
      if(!response.ok){throw new Error(await response.text())}
      const c=await response.json();$('cameraBadge').textContent=c.resolution+' · '+c.width+'×'+c.height;actionState.textContent='参数已应用；实时流正在重连。';connectStream('');
    }catch(e){actionState.textContent='应用失败：'+e.message}
    finally{button.disabled=false}
  };
  function downloadBlob(blob,name){const a=document.createElement('a');const url=URL.createObjectURL(blob);a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000)}
  $('capture').onclick=async()=>{
    const button=$('capture');button.disabled=true;actionState.textContent='正在抓拍并绑定距离...';
    try{
      const response=await fetch('/api/snapshot',{cache:'no-store'});if(!response.ok){throw new Error(await response.text())}
      const image=await response.blob();const metadataText=response.headers.get('X-Capture-Metadata');if(!metadataText){throw new Error('设备未返回抓拍元数据')}
      const metadata=JSON.parse(metadataText);const stamp=new Date().toISOString().replace(/[:.]/g,'-');const base='atoms3r_'+stamp;
      metadata.browser_downloaded_at=new Date().toISOString();metadata.image_file=base+'.jpg';
      downloadBlob(image,metadata.image_file);downloadBlob(new Blob([JSON.stringify(metadata,null,2)],{type:'application/json'}),base+'.json');
      actionState.textContent='已请求下载 '+metadata.image_file+' 和配套 JSON。';
    }catch(e){actionState.textContent='抓拍失败：'+e.message}
    finally{button.disabled=false}
  };
  async function updateRange(){
    try{
      const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),1800);
      const r=await fetch('/api/range',{cache:'no-store',signal:controller.signal});clearTimeout(timeout);if(!r.ok){throw new Error('HTTP '+r.status)}
      const d=await r.json();rangeFailures=0;
      if(d.valid){distance.textContent=d.range_m.toFixed(3)+' m';reading.className='reading valid';state.textContent='测距有效 · '+d.age_ms+' ms'}
      else{distance.textContent=d.status==='NOT_READY'?'等待传感器':'无有效距离';reading.className='reading invalid';state.textContent=d.status}
    }catch(e){rangeFailures++;distance.textContent='连接中断';reading.className='reading invalid';state.textContent='测距重连中 · '+rangeFailures;showNotice('距离接口暂时不可用，网页正在自动重试。')}
    setTimeout(updateRange,rangeFailures?Math.min(1000*rangeFailures,5000):200);
  }
  async function updateStatus(){
    try{
      const s=await (await fetch('/api/status',{cache:'no-store'})).json();statusFailures=0;
      $('fpsBadge').textContent=s.camera.recent_fps.toFixed(1)+' fps';
      if(s.camera.recent_fps>0){lastHealthyStream=Date.now()}
      else if(!staticMode&&Date.now()-lastHealthyStream>6000){lastHealthyStream=Date.now();connectStream('检测到画面停滞，正在自动重连...')}
    }catch(e){statusFailures++;if(statusFailures>2){showNotice('设备状态连接中断，页面会继续自动重试。')}}
    setTimeout(updateStatus,statusFailures?3000:1500);
  }
  window.addEventListener('load',()=>{loadCamera();updateRange();updateStatus();setTimeout(()=>connectStream(''),150)});
</script></body></html>
)HTML";

constexpr char kStatusHtml[] PROGMEM = R"HTML(
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AtomS3R 设备状态</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:#07111f;color:#f4f7fb;padding:16px}main{width:min(100%,900px);margin:auto}.top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}h1{font-size:28px}.back{color:#8cdbff;text-decoration:none}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.card{background:#0d1b2b;border:1px solid #ffffff18;border-radius:16px;padding:17px}.card h2{font-size:18px;margin:0 0 12px;color:#a9ddff}.row{display:flex;justify-content:space-between;gap:14px;padding:7px 0;border-bottom:1px solid #ffffff0e}.row:last-child{border:0}.value{text-align:right;color:#63ffb4;overflow-wrap:anywhere}.bad{color:#ffbd78}.notice{margin:14px 0;padding:11px 14px;border-radius:11px;background:#50291c;color:#ffd7bd;display:none}.notice.show{display:block}small{color:#8fa2b7}
</style></head><body><main><div class="top"><h1>设备状态</h1><a class="back" href="/">← 返回实时画面</a></div><div id="notice" class="notice"></div>
<div class="grid">
<section class="card"><h2>系统</h2><div class="row"><span>固件</span><span class="value" id="firmware">--</span></div><div class="row"><span>运行时间</span><span class="value" id="uptime">--</span></div><div class="row"><span>空闲内存</span><span class="value" id="heap">--</span></div></section>
<section class="card"><h2>Wi-Fi</h2><div class="row"><span>连接</span><span class="value" id="wifi">--</span></div><div class="row"><span>IP</span><span class="value" id="ip">--</span></div><div class="row"><span>信号</span><span class="value" id="rssi">--</span></div><div class="row"><span>重连尝试</span><span class="value" id="reconnects">--</span></div></section>
<section class="card"><h2>相机</h2><div class="row"><span>状态</span><span class="value" id="camera">--</span></div><div class="row"><span>参数</span><span class="value" id="cameraConfig">--</span></div><div class="row"><span>近期帧率</span><span class="value" id="fps">--</span></div><div class="row"><span>流客户端</span><span class="value" id="clients">--</span></div><div class="row"><span>累计发送</span><span class="value" id="frames">--</span></div></section>
<section class="card"><h2>ToF4M</h2><div class="row"><span>传感器</span><span class="value" id="tof">--</span></div><div class="row"><span>测量状态</span><span class="value" id="rangeStatus">--</span></div><div class="row"><span>距离</span><span class="value" id="range">--</span></div><div class="row"><span>样本年龄</span><span class="value" id="age">--</span></div></section>
</div><p><small>状态每 1 秒刷新。ToF4M 仍是中央单区测距；本页不表示整幅深度或安全状态。</small></p>
</main><script>
const $=id=>document.getElementById(id);let failures=0;
function duration(ms){const s=Math.floor(ms/1000),d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return(d?d+'天 ':'')+h+'小时 '+m+'分 '+s%60+'秒'}
async function update(){
 try{
  const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok){throw new Error('HTTP '+r.status)}const s=await r.json();failures=0;$('notice').className='notice';
  $('firmware').textContent=s.firmware_version;$('uptime').textContent=duration(s.uptime_ms);$('heap').textContent=Math.round(s.free_heap_bytes/1024)+' KiB';
  $('wifi').textContent=s.wifi.connected?'已连接':'已断开';$('wifi').className='value '+(s.wifi.connected?'':'bad');$('ip').textContent=s.wifi.ip||'--';$('rssi').textContent=s.wifi.connected?s.wifi.rssi_dbm+' dBm':'--';$('reconnects').textContent=s.wifi.reconnect_attempts;
  $('camera').textContent=s.camera.ready?'正常':'异常';$('camera').className='value '+(s.camera.ready?'':'bad');$('cameraConfig').textContent=s.camera.resolution+' '+s.camera.width+'×'+s.camera.height+' / Q'+s.camera.jpeg_quality;$('fps').textContent=s.camera.recent_fps.toFixed(1)+' fps';$('clients').textContent=s.camera.stream_clients;$('frames').textContent=s.camera.total_frames;
  $('tof').textContent=s.tof.ready?'正常':'异常';$('tof').className='value '+(s.tof.ready?'':'bad');$('rangeStatus').textContent=s.tof.status;$('range').textContent=s.tof.valid?s.tof.range_mm+' mm':'无有效距离';$('age').textContent=s.tof.age_ms+' ms';
 }catch(e){failures++;$('notice').textContent='状态连接失败（第 '+failures+' 次），正在自动重试：'+e.message;$('notice').className='notice show'}
 setTimeout(update,failures?Math.min(1000*failures,5000):1000)
}update();
</script></body></html>
)HTML";

constexpr char kSetupHtml[] PROGMEM = R"HTML(
<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>配置 AtomS3R Wi-Fi</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:linear-gradient(145deg,#07111f,#102b42);color:#f5f8fb;padding:18px}
main{width:min(100%,430px);background:#0b1726e8;border:1px solid #ffffff22;border-radius:20px;padding:24px;box-shadow:0 24px 70px #0008}h1{font-size:25px;margin:0 0 8px}p{color:#afc0d2;line-height:1.55}label{display:block;margin:15px 0 6px;font-weight:650}input{width:100%;padding:12px;border:1px solid #ffffff33;border-radius:10px;background:#07111f;color:#fff;font-size:16px}button{width:100%;margin-top:20px;padding:13px;border:0;border-radius:11px;background:#38dc91;color:#032114;font-size:16px;font-weight:750}small{display:block;margin-top:14px;color:#8296aa;line-height:1.5}
</style></head><body><main><h1>连接你的 Wi-Fi</h1><p>输入 2.4 GHz Wi-Fi 信息。凭据只保存在这台 Atom 的本地 NVS，不会通过串口输出或写入代码仓库。</p>
<form method="post" action="/save"><label for="ssid">Wi-Fi 名称</label><input id="ssid" name="ssid" maxlength="32" required autocomplete="off">
<label for="password">Wi-Fi 密码</label><input id="password" name="password" type="password" maxlength="63" autocomplete="new-password">
<button type="submit">保存并连接</button></form><small>保存后 Atom 会重启并关闭临时热点。手机/电脑重新连接原 Wi-Fi，然后打开 http://atoms3r-tof.local/ 。配置页面使用局域网 HTTP，请只在可信环境操作。</small></main></body></html>
)HTML";

constexpr char kSavedHtml[] PROGMEM = R"HTML(
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>已保存</title>
<style>:root{color-scheme:dark;font-family:system-ui}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#07111f;color:#fff;text-align:center;padding:24px}main{max-width:520px}h1{color:#63ffb4}p{line-height:1.6;color:#b8c7d8}</style></head>
<body><main><h1>Wi-Fi 已保存</h1><p>Atom 正在重启并连接你的网络。请把手机或电脑切回原 Wi-Fi，约 20 秒后访问：<br><strong>http://atoms3r-tof.local/</strong></p></main></body></html>
)HTML";
