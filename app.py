"""Pindi Studio launcher: shows an animated splash screen while the AI engine loads,
then opens the main window. The real app lives in pindi_core.py."""
import json
import os
import sys
import threading
import time
import traceback

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ.setdefault(
    "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
    "--enable-gpu-rasterization --enable-zero-copy --ignore-gpu-blocklist --enable-smooth-scrolling --autoplay-policy=no-user-gesture-required",
)

# In a windowed .exe, stdout/stderr are None, which crashes some libraries
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import webview

try:
    webview.settings["ALLOW_DOWNLOADS"] = True  # lets files received in Chat be saved
except Exception:
    pass

APP_NAME = "Pindi Studio"
MIN_SPLASH_SECONDS = 3.9  # intro: star, name reveal, colour wipe, with sound

SPLASH_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pindi Studio</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#f6f1e7;font-family:Roboto,"Segoe UI",Arial,sans-serif;user-select:none;-webkit-user-select:none}
body{position:relative;color:#1f2a28;transition:opacity .55s ease,transform .55s ease}
body.exit{opacity:0;transform:scale(1.06)}
/* gradient colour wipe */
#wipe{position:absolute;inset:0;background:linear-gradient(180deg,#14958b 0%,#0f766e 45%,#094a45 100%);clip-path:circle(0% at 50% 50%);animation:wipe .75s cubic-bezier(.65,0,.25,1) 2.25s forwards}
@keyframes wipe{to{clip-path:circle(80% at 50% 50%)}}
#glow{position:absolute;left:50%;top:50%;width:420px;height:420px;margin:-210px 0 0 -210px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,122,89,.30),rgba(15,118,110,.12) 45%,transparent 70%);opacity:0;animation:glow 1.6s ease .1s forwards}
@keyframes glow{to{opacity:1}}
#lock{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}
#word{overflow:hidden;white-space:nowrap;max-width:0;opacity:0;font-size:52px;font-weight:900;font-style:italic;letter-spacing:1px;padding:0 2px;
  animation:word .65s cubic-bezier(.2,.8,.2,1) 1.55s forwards;transition:color .4s ease .25s}
#word i{font-style:italic;color:#ff7a59;transition:color .4s ease .25s}
@keyframes word{from{max-width:0;opacity:0}to{max-width:520px;opacity:1}}
#star{order:2;margin-left:12px;width:140px;height:140px;flex:none;opacity:0;transform:scale(.15) rotate(-240deg);
  animation:starin .7s cubic-bezier(.2,.9,.3,1.2) .1s forwards,pulse 1.0s ease-in-out .85s,shrink .65s cubic-bezier(.2,.8,.2,1) 1.55s forwards}
@keyframes starin{to{opacity:1;transform:scale(1) rotate(0)}}
@keyframes pulse{0%{transform:scale(1) rotate(0)}35%{transform:scale(.8) rotate(120deg)}70%{transform:scale(1.15) rotate(240deg)}100%{transform:scale(1) rotate(360deg)}}
@keyframes shrink{from{width:140px;height:140px;opacity:1;transform:scale(1) rotate(0)}to{width:62px;height:62px;opacity:1;transform:scale(1) rotate(360deg)}}
body.wiped #word,body.wiped #word i{color:#fff}
#star svg{width:100%;height:100%;display:block;transition:filter .4s ease .25s}
#tag{position:absolute;left:0;right:0;top:calc(50% + 52px);text-align:center;font-size:13px;font-weight:500;letter-spacing:6px;color:#fff;opacity:0;animation:tag .6s ease 3.0s forwards}
@keyframes tag{to{opacity:.85}}
#status{position:absolute;left:0;right:0;bottom:16px;text-align:center;font-size:12px;letter-spacing:1.5px;opacity:0;color:#5f6b68;animation:statin .6s ease 1.2s forwards;transition:color .3s}
body.wiped #status{color:#e0f3f0}
@keyframes statin{to{opacity:.8}}
.close{position:absolute;top:10px;right:14px;font-size:18px;opacity:.4;cursor:pointer;z-index:10;transition:color .3s}
body.wiped .close{color:#fff}.close:hover{opacity:1}
#err{display:none;position:absolute;left:50%;bottom:44px;width:560px;max-width:92vw;margin-left:-280px;padding:10px 12px;font-size:11px;text-align:left;background:rgba(220,38,38,.12);border:1px solid rgba(220,38,38,.5);border-radius:8px;white-space:pre-wrap;max-height:150px;overflow:auto;color:#fff;background:rgba(120,20,20,.85);z-index:11}
</style></head><body>
<div id="glow"></div>
<div id="wipe"></div>
<div id="lock">
  <div id="star"><svg viewBox="30 10 340 340"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffb29a"/><stop offset=".5" stop-color="#ff7a59"/><stop offset="1" stop-color="#0f766e"/></linearGradient></defs><path d="M200 200 C150 150 150 70 200 30 C250 70 250 150 200 200Z" fill="url(#g)" opacity=".95" transform="rotate(0 200 200)"/><path d="M200 200 C150 150 150 70 200 30 C250 70 250 150 200 200Z" fill="url(#g)" opacity=".95" transform="rotate(120 200 200)"/><path d="M200 200 C150 150 150 70 200 30 C250 70 250 150 200 200Z" fill="url(#g)" opacity=".95" transform="rotate(240 200 200)"/><circle cx="200" cy="200" r="16" fill="#fff"/></svg></div>
  <div id="word">KHOKHAR <i>333</i></div>
</div>
<div id="tag">PINDI STUDIO</div>
<div class="close" onclick="closeMe()">&#10005;</div>
<div id="status">STARTING...</div>
<div id="err"></div>
<script>
(function(){
  /* the star turns white once the colour wipe covers the screen */
  setTimeout(function(){document.body.classList.add('wiped');document.querySelector('#star svg').style.filter='brightness(0) invert(1)'},2650);
  /* original synthesized intro sound (no audio file needed) */
  try{
    var AC=window.AudioContext||window.webkitAudioContext;if(!AC)return;
    var ac=new AC(),out=ac.createGain();out.gain.value=.55;out.connect(ac.destination);
    var comp=ac.createDynamicsCompressor();out.disconnect();out.connect(comp);comp.connect(ac.destination);
    function play(){
      var t=ac.currentTime+.05;
      function tone(type,f1,f2,t0,dur,vol,att){var o=ac.createOscillator(),g=ac.createGain();o.type=type;
        o.frequency.setValueAtTime(f1,t+t0);if(f2!==f1)o.frequency.exponentialRampToValueAtTime(f2,t+t0+dur);
        g.gain.setValueAtTime(.0001,t+t0);g.gain.exponentialRampToValueAtTime(vol,t+t0+(att||.02));g.gain.exponentialRampToValueAtTime(.0001,t+t0+dur);
        o.connect(g);g.connect(out);o.start(t+t0);o.stop(t+t0+dur+.05)}
      /* rising sparkle while the star spins */
      tone('sine',420,1500,0.05,0.8,.12,.3);tone('triangle',840,2400,0.1,0.7,.05,.3);
      [1320,1760,2093].forEach(function(f,i){tone('sine',f,f,.15+i*.2,.35,.05,.01)});
      /* low impact as the name appears */
      tone('sine',110,38,1.5,1.4,.5,.01);tone('sine',55,40,1.5,1.4,.35,.01);
      /* warm chord swell, A major-ish */
      [220,277.2,329.6,440].forEach(function(f){tone('triangle',f,f,1.5,1.9,.07,.12)});
      /* whoosh for the colour wipe */
      var n=ac.createBufferSource(),len=ac.sampleRate*1,b=ac.createBuffer(1,len,ac.sampleRate),d=b.getChannelData(0);
      for(var i=0;i<len;i++)d[i]=Math.random()*2-1;n.buffer=b;
      var bp=ac.createBiquadFilter();bp.type='bandpass';bp.Q.value=1.2;bp.frequency.setValueAtTime(300,t+2.1);bp.frequency.exponentialRampToValueAtTime(5000,t+2.9);
      var ng=ac.createGain();ng.gain.setValueAtTime(.0001,t+2.1);ng.gain.exponentialRampToValueAtTime(.35,t+2.5);ng.gain.exponentialRampToValueAtTime(.0001,t+3.1);
      n.connect(bp);bp.connect(ng);ng.connect(out);n.start(t+2.1);
      /* bright finish chime */
      tone('sine',880,880,2.7,1.0,.14,.01);tone('sine',1318.5,1318.5,2.75,1.0,.08,.01);tone('sine',1760,1760,2.8,1.0,.05,.01);
    }
    var started=false;function go(){if(started)return;started=true;if(ac.state==='suspended'){ac.resume().then(play,play)}else play()}
    go();
    /* if the engine blocked autoplay, start the sound on the first click */
    setTimeout(function(){if(ac.state!=='running'){started=false;addEventListener('pointerdown',go,{once:true})}},300);
  }catch(e){}
})();
function setStatus(t){document.getElementById('status').textContent=String(t).toUpperCase()}
function exitSlide(){document.body.classList.add('exit')}
function closeMe(){if(window.pywebview&&window.pywebview.api){window.pywebview.api.close()}}
function showError(t){
  var e=document.getElementById('err');e.style.display='block';e.textContent=t;
  document.getElementById('status').textContent='COULD NOT START - DETAILS BELOW';
}
</script>
</body></html>
"""

def _icon_path():
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for base in (Path(getattr(sys, "_MEIPASS", here)), here, Path(sys.executable).parent):
        for icon_name in ("pindi.ico", "pindi_icon.ico", "pindi.ico.ico"):
            p = base / icon_name
            if p.is_file():
                return str(p)
        try:  # any other icon whose name starts with "pindi"
            found = sorted(base.glob("pindi*.ico"))
            if found:
                return str(found[0])
        except OSError:
            pass
    return None


def brand_taskbar():
    """Windows: give the app its own identity, so the taskbar does not group it under python.exe."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Khokhar.PindiStudio.1")
    except Exception:
        pass


def set_window_icon(title, tries=40):
    """Windows: put the P icon (pindi.ico) on the window with this title, so the taskbar shows it
    instead of the Python icon. When the app runs as the .exe, the exe's own icon is used already."""
    if os.name != "nt":
        return
    path = _icon_path()
    print("[icon] file:", path, flush=True)
    if not path:
        return
    try:
        import ctypes
        from ctypes import c_int, c_uint, c_void_p, c_wchar_p

        u = ctypes.windll.user32
        u.FindWindowW.argtypes = [c_void_p, c_wchar_p]
        u.FindWindowW.restype = c_void_p
        u.LoadImageW.argtypes = [c_void_p, c_wchar_p, c_uint, c_int, c_int, c_uint]
        u.LoadImageW.restype = c_void_p
        u.SendMessageW.argtypes = [c_void_p, c_uint, c_void_p, c_void_p]
        applied = 0
        for _ in range(tries):
            hwnd = u.FindWindowW(None, title)
            if hwnd:
                big = u.LoadImageW(None, path, 1, 0, 0, 0x10 | 0x40)  # IMAGE_ICON, from file, default size
                small = u.LoadImageW(None, path, 1, 16, 16, 0x10)
                if big:
                    u.SendMessageW(hwnd, 0x80, 1, big)    # WM_SETICON, ICON_BIG
                if small:
                    u.SendMessageW(hwnd, 0x80, 0, small)  # ICON_SMALL
                applied += 1
                print(f"[icon] window '{title}' found, icon set ({'ok' if big else 'FAILED to load the file'})", flush=True)
                if applied >= 4:  # set it a few times: the window can reset its icon while it opens / maximizes
                    return
                time.sleep(2.0)
                continue
            time.sleep(0.5)
    except Exception as e:
        print("[icon] error:", e, flush=True)


def set_native_icon(window, delays=(1.0, 3.0, 6.0)):
    """Second way: set the icon on the pywebview window's own .NET form."""
    path = _icon_path()
    if os.name != "nt" or not path:
        return
    for d in delays:
        time.sleep(d)
        try:
            from System import Action
            from System.Drawing import Icon

            form = window.native
            ico = Icon(path)

            def apply():
                form.Icon = ico

            if form.InvokeRequired:
                form.Invoke(Action(apply))
            else:
                apply()
            print("[icon] .NET window icon set", flush=True)
        except Exception as e:
            print("[icon] .NET way failed:", repr(e), flush=True)
            return


def apply_updates():
    """Small updates without rebuilding the .exe: if an 'updates' folder next to the app holds
    pindi_chat.py and/or pindi_core.py, those files are used instead of the ones packed inside."""
    import importlib.util
    from pathlib import Path

    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    folder = base / "updates"
    used = []
    for name in ("pindi_chat", "pindi_core"):  # chat first: the core imports it
        file = folder / f"{name}.py"
        if not file.is_file():  # an update file made before the rename is still accepted
            file = folder / (name.replace("pindi_", "zynth_") + ".py")
        if not file.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, str(file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            used.append(name)
        except BaseException:
            sys.modules.pop(name, None)
            print("Could not load the update", file, file=sys.stderr)
            traceback.print_exc()
            return used, traceback.format_exc()
    return used, None


_splash = None


class SplashApi:
    def close(self):
        if _splash is not None:
            _splash.destroy()


def load_and_show(splash):
    started = time.time()
    threading.Thread(target=set_window_icon, args=(f"{APP_NAME} Loading",), daemon=True).start()
    threading.Thread(target=set_native_icon, args=(splash, (0.5, 2.0)), daemon=True).start()

    def js(code):
        try:
            splash.evaluate_js(code)
        except Exception:
            pass

    try:
        js('setStatus("Loading AI engine (the first start can take a minute)...")')
        _, update_error = apply_updates()
        if update_error:
            js("setStatus(" + json.dumps("Update file has an error - using the built-in version") + ")")
            for name in ("pindi_chat", "pindi_core"):
                sys.modules.pop(name, None)
        import pindi_core

        js('setStatus("Starting studio...")')
        url = pindi_core.start_server()
        js('setStatus("Ready")')

        # keep the splash on screen long enough to be seen
        time.sleep(max(0.0, MIN_SPLASH_SECONDS - (time.time() - started)))
        js("exitSlide()")  # the Z slides away to the right
        time.sleep(0.85)
        main = webview.create_window(APP_NAME, url, width=1280, height=800, min_size=(800, 600), maximized=True)
        threading.Thread(target=set_window_icon, args=(APP_NAME,), daemon=True).start()
        threading.Thread(target=set_native_icon, args=(main,), daemon=True).start()

        def fill_screen():  # some PCs ignore maximized=True at creation, so ask again once the window is shown
            for _ in range(3):
                time.sleep(0.7)
                try:
                    main.maximize()
                except Exception:
                    pass

        try:
            main.events.shown += lambda: threading.Thread(target=fill_screen, daemon=True).start()
        except Exception:
            threading.Thread(target=fill_screen, daemon=True).start()
        time.sleep(0.4)
        splash.destroy()
    except BaseException:
        err = traceback.format_exc()
        try:
            print(err, file=sys.stderr, flush=True)
        except Exception:
            pass
        js("showError(" + json.dumps(err[-1500:]) + ")")


def main():
    global _splash
    brand_taskbar()
    _splash = webview.create_window(
        f"{APP_NAME} Loading",
        html=SPLASH_HTML,
        width=800,
        height=450,
        frameless=True,
        resizable=False,
        on_top=True,
        js_api=SplashApi(),
    )
    icon = _icon_path()
    try:
        webview.start(load_and_show, _splash, icon=icon) if icon else webview.start(load_and_show, _splash)
    except TypeError:  # an older pywebview without the icon option
        webview.start(load_and_show, _splash)
    # All windows are closed: close the whole app
    os._exit(0)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
