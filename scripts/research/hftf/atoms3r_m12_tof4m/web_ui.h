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
    main{width:min(100%,980px);margin:auto;padding:16px}.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
    h1{font-size:clamp(20px,4vw,30px);margin:4px 0}.pill{padding:7px 11px;border-radius:999px;background:#17304c;color:#bfe4ff;font-size:14px}
    .viewer{position:relative;margin-top:14px;background:#000;border-radius:18px;overflow:hidden;aspect-ratio:4/3;box-shadow:0 18px 60px #0008}
    #stream{width:100%;height:100%;object-fit:contain;display:block}.reticle{position:absolute;left:39%;top:35%;width:22%;height:30%;border:3px solid #43f5a3;border-radius:14px;box-shadow:0 0 0 9999px #0002,0 0 18px #43f5a399;pointer-events:none}
    .reading{position:absolute;left:50%;bottom:7%;transform:translateX(-50%);min-width:180px;text-align:center;background:#06121ddd;border:1px solid #ffffff38;border-radius:14px;padding:10px 18px;backdrop-filter:blur(8px)}
    #distance{font-size:clamp(30px,7vw,54px);font-weight:750;line-height:1.05}.label{font-size:13px;color:#b8c7d8;margin-top:5px}.invalid #distance{color:#ffc15a}.valid #distance{color:#63ffb4}
    .foot{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-top:14px;color:#9eb0c4;font-size:14px}.foot p{margin:0;max-width:700px}
    button{border:1px solid #ffffff33;border-radius:10px;background:#17283b;color:#eef6ff;padding:9px 13px;cursor:pointer}
  </style>
</head>
<body><main>
  <div class="top"><h1>AtomS3R 实时画面＋距离</h1><span class="pill">XGA · 1024×768</span><span class="pill" id="state">正在连接测距...</span></div>
  <div class="viewer"><img id="stream" alt="AtomS3R-M12 实时摄像头画面">
    <div class="reticle" aria-hidden="true"></div>
    <div class="reading invalid" id="reading"><div id="distance">--</div><div class="label">中央单点 ToF 距离</div></div>
  </div>
  <div class="foot"><p>绿色框仅表示 ToF4M 的中央窄视场；数值不代表整幅图的深度，也不能作为独立安全判断。</p>
    <form method="post" action="/forget" onsubmit="return confirm('清除 Wi-Fi 配置并重新进入设置模式？')"><button type="submit">更换 Wi-Fi</button></form>
  </div>
</main><script>
  const img=document.getElementById('stream');
  window.addEventListener('load',()=>{img.src='http://'+location.hostname+':81/stream';});
  const reading=document.getElementById('reading'),distance=document.getElementById('distance'),state=document.getElementById('state');
  async function update(){
    try{
      const r=await fetch('/api/range',{cache:'no-store'}); const d=await r.json();
      if(d.valid){distance.textContent=d.range_m.toFixed(3)+' m';reading.className='reading valid';state.textContent='测距有效 · '+d.age_ms+' ms';}
      else{distance.textContent=d.status==='NOT_READY'?'等待传感器':'无有效距离';reading.className='reading invalid';state.textContent=d.status;}
    }catch(e){distance.textContent='连接中断';reading.className='reading invalid';state.textContent='正在重连...';}
  }
  update(); setInterval(update,200);
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
