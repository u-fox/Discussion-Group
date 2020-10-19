var banner = {
  htmlMem: "",
  loopCount: 0
};

/////////////////////////////////////////////////////////
//
// Query Selectors
//
/////////////////////////////////////////////////////////
function $(el) {
  var elements = ArrayFrom(document.querySelectorAll(el));
  if (elements.length > 1) {
    return elements;
  }
  if (elements.length == 1) {
    return elements[0];
  }
}

function show(el) {
  set(el, { display: "block" });
}
function hide(el) {
  set(el, { display: "none" });
}

/////////////////////////////////////////////////////////
//
// Listeners
//
/////////////////////////////////////////////////////////
function addListeners() {
  console.log(">> LISTENERS ADDED");
  $(".clickthrough").addEventListener("click", onClickThrough);
  $(".clickthrough").addEventListener("mouseover", onMouseOver);
  $(".clickthrough").addEventListener("mouseout", onMouseLeave);

  // include replay button, if replay button exist
  if (document.querySelectorAll(".replay").length > 0) {
    $(".replay").addEventListener("click", reset);
  }
}

function onClickThrough() {
  window.open(window.clickTag);
  // Enabler.exit('Backgrond Clickthrough');
  console.log(">> Clickthrough");
}

function onMouseOver() {
  // to(".cta", 0.3, { scale: 1.05 }, "out");
  // to(".cta .button-fill", 0.3, { fill: "#e53a94" }, "out");
}
function onMouseLeave() {
  // to(".cta", 0.3, { scale: 1.0 }, "out");
  // to(".cta", 0.3, { x: 0, y: 0 }, "out");
  // to(".cta .button-fill", 0.3, { fill: "#3c57a6" }, "out");
}

/////////////////////////////////////////////////////////
//
// Polyfills
//
/////////////////////////////////////////////////////////
ArrayFrom = function (list) {
  var array = [];
  for (i = 0; i < list.length; i++) {
    array.push(list[i]);
  }
  return array;
};

/////////////////////////////////////////////////////////
//
// Image Preloader
//
/////////////////////////////////////////////////////////
function ImagePreloader(images, callback) {
  this.images = images;
  this.callback = callback;
  this.imagesLoaded = 0;
  console.log(this);
  var THIS = this;
  this.images.map(function (image, i) {
    THIS.loadImage(image);
  });
}

ImagePreloader.prototype.loadImage = function (src) {
  var THIS = this;
  var newImage = new Image();
  newImage.src = src;
  newImage.onload = function (e) {
    THIS.imagesLoaded++;
    if (THIS.imagesLoaded == THIS.images.length) {
      console.log(">> ALL IMAGES LOADED");
      THIS.callback();
    }
  };
};

////////////////////////////////////////////////////////////////////////
//
// Tween Nano
//
////////////////////////////////////////////////////////////////////////
var tweenLog = [];

tween = function (_functionType, _element, _time, _object, _ease, _delay) {
  if (_element.style) {
    // console.log(_element)
    if (_element.style.display == "none") _element.style.display = "block";
  } else if ($(_element).style) {
    // console.log($(_element))
    if ($(_element).style.display != "block" && $(_element).style.display != "inline-block")
      $(_element).style.display = "block";
  }

  var easeType = "Power0.easeIn";
  if (_ease) _ease = _ease.toLowerCase();
  switch (_ease) {
    case "in":
      easeType = "Power2.easeIn";
      break;
    case "out":
      easeType = "Power2.easeOut";
      break;
    case "inout":
      easeType = "Power2.easeInOut";
      break;
    case "sinein":
      easeType = "Sine.easeIn";
      break;
    case "sineout":
      easeType = "Sine.easeOut";
      break;
    case "sineinout":
      easeType = "Sine.easeInOut";
      break;
    case "backout":
      easeType = "Back.easeOut";
      break;
    case "override":
      break; // for custom tweens
    case "":
      break; // for custom tweens
    default:
      break;
  }
  if (_ease != "override") _object.ease = easeType;
  _object.delay = _delay * timeScale;
  _functionType(_element, _time * timeScale, _object);
  tweenLog.push(_element);
};
var timeScale = 1;
function setTweenTime(n) {
  timeScale = n;
}
set = TweenMax.set;
from = function (_element, _time, _object, _ease, _delay) {
  tween(TweenMax.from, _element, _time, _object, _ease, _delay);
};
to = function (_element, _time, _object, _ease, _delay) {
  tween(TweenMax.to, _element, _time, _object, _ease, _delay);
};
wait = function (_time, _function, _params) {
  TweenMax.delayedCall(_time * timeScale, _function, _params);
  tweenLog.push(_function);
};

// Kill specific tween
kill = function (_element) {
  TweenMax.killTweensOf(_element);
};

// Kill all tweens
killAll = function () {
  TweenMax.killAll();
};

/////////////////////////////////////////////////////////
//
// META DATA EXTRACTOR
//
/////////////////////////////////////////////////////////
// returns an object with width & height of banner.
function getAdSizeMeta() {
  var content;
  var metaTags = document.getElementsByTagName("meta");
  ArrayFrom(metaTags).map(function (meta) {
    if (meta.name == "ad.size") {
      content = meta.content.split(",");
    }
  });
  var size = {
    width: parseInt(content[0].substring(6)),
    height: parseInt(content[1].substring(7))
  };
  return size;
}

// SET BANNER SIZE --------------------------------------------
function setBannerSize() {
  // check banner exists.
  if (!window.hasOwnProperty("banner")) window.banner = {};

  banner.width = getAdSizeMeta().width;
  banner.height = getAdSizeMeta().height;
  $(".banner").style.width = banner.width + "px";
  $(".banner").style.height = banner.height + "px";

  document.title = getAdSizeMeta().width + "x" + getAdSizeMeta().height;
}

////////////////////////////////////////////////////////////////////////////////
//
//  Detect IE
//
////////////////////////////////////////////////////////////////////////////////
function isIE(userAgent) {
  userAgent = userAgent || navigator.userAgent;
  return userAgent.indexOf("MSIE ") > -1 || userAgent.indexOf("Trident/") > -1 || userAgent.indexOf("Edge/") > -1;
}
function checkForIE() {
  if (isIE()) {
    document.querySelector(".banner").classList.add("IE-detected");
  }
}
checkForIE();

////////////////////////////////////////////////////////////////////////////////
//
//  Lines in / Lines out
//
////////////////////////////////////////////////////////////////////////////////
function linesIn(frame, tween, delay, delayArray) {
  if (!tween) tween = { x: -20, alpha: 0 };
  if (!delay) delay = 0;
  var delayBetweenTweens = 0.05;
  var ease = "out";
  var manualDelay = 0;

  ArrayFrom(document.querySelectorAll(frame + " .line")).map(function (line, i) {
    if (delayArray && delayArray[i]) manualDelay += delayArray[i];
    from(line, 0.5, tween, ease, delayBetweenTweens * i + delay + manualDelay);
  });
}
function linesOut(frame, tween, delay) {
  if (!tween) tween = { x: 20, alpha: 0 };
  if (!delay) delay = 0;
  var delayBetweenTweens = 0.05;
  var ease = "in";

  ArrayFrom(document.querySelectorAll(frame + " .line")).map(function (line, i) {
    to(line, 0.4, tween, ease, delayBetweenTweens * i + delay);
  });
}

// TERMS AND CONTITINOS PANEL --------------------------------------
function showTerms() {
  kill(loop);
  if (!banner.looping) {
    banner.termsShowing = true;

    // remove / add listeners
    $(".terms-button").removeEventListener("click", showTerms);
    wait(0.3, function () {
      $(".terms-panel .hitarea").addEventListener("click", hideTerms);
      $(".terms-close").addEventListener("click", hideTerms);
    });

    // move terms
    to(".terms-button", 0.3, { alpha: 0 }, "in");
    show(".terms-panel");
    set(".terms-panel .panel", { y: "100%" });
    to(".terms-panel .panel", 0.3, { y: "0%" }, "inOut");
    set(".terms-panel", { alpha: 0 });
    to(".terms-panel", 0.3, { alpha: 1 }, "out");
  }
}

function hideTerms(onReplay) {
  // remove / add listeners
  $(".terms-panel .hitarea").removeEventListener("click", hideTerms);
  $(".terms-close").removeEventListener("click", hideTerms);
  wait(0.3, function () {
    $(".terms-button").addEventListener("click", showTerms);
  });

  // move terms
  banner.termsShowing = false;
  to(".terms-button", 0.3, { alpha: 1 }, "in");
  show(".terms-panel");
  to(".terms-panel", 0.3, { alpha: 0 }, "in");
  to(".terms-panel .panel", 0.3, { y: "100%" }, "inOut");

  wait(0.3, function () {
    hide(".terms-panel");
  });

  if (!onReplay) {
    wait(4, loop);
  }
}

///////////////////////////////////////////////////////
//
// ---- SPLIT SEPARATE FRAMES ----
//
///////////////////////////////////////////////////////
function separateFrames() {
  killAll(), (banner.width = $(".banner").offsetWidth), (banner.height = $(".banner").offsetHeight);
  var t = [],
    a = "";
  ArrayFrom($(".frame")).map(function (e, n) {
    /frame[0-9]/.test(e.className) && (t.push(e), (a += htmlOriginal));
  }),
    ($("body").innerHTML = a);
  var i = document.querySelectorAll(".banner");
  ArrayFrom(i).map(function (e, n) {
    var t = n + 1,
      a = i[n];
    t == i.length ? (a.className += " endframe banner" + t) : (a.className += " banner" + t),
      set(a, { position: "relative", margin: "0 15px 30px 15px", transform: "scale(1)", display: "inline-block" }),
      ($(".banner" + t + " .frame" + t).style.display = "block"),
      (banner.expanded = { height: a.getBoundingClientRect().top + banner.height + 50 }),
      (t = n + 0);
    var r = a.getBoundingClientRect().left - document.querySelector(".banner1").getBoundingClientRect().left,
      l = a.getBoundingClientRect().top - document.querySelector(".banner1").getBoundingClientRect().top;
    from(a, 0.3, { x: -r, y: -l, alpha: 0 }, "out");
  }),
    onSplitFrames();
}
function resetFrames() {
  killAll();
  var e = document.querySelectorAll(".banner");
  ArrayFrom(e).map(function (e, n) {
    var t = e.getBoundingClientRect().left - $(".banner1").getBoundingClientRect().left,
      a = e.getBoundingClientRect().top - $(".banner1").getBoundingClientRect().top;
    to(e, 0.2, { x: -t, y: -a, alpha: 0 }, "in");
  }),
    wait(0.2, function () {
      killAll(), ($("body").innerHTML = htmlOriginal), show(".banner"), loop();
    });
}


/*!
 * VERSION: 0.0.5
 * DATE: 2015-05-19
 * UPDATES AND DOCS AT: http://greensock.com
 *
 * @license Copyright (c) 2008-2015, GreenSock. All rights reserved.
 * DrawSVGPlugin is a Club GreenSock membership benefit; You must have a valid membership to use
 * this code without violating the terms of use. Visit http://greensock.com/club/ to sign up or get more details.
 * This work is subject to the software agreement that was issued with your membership.
 * 
 * @author: Jack Doyle, jack@greensock.com
 */
var _gsScope="undefined"!=typeof module&&module.exports&&"undefined"!=typeof global?global:this||window;(_gsScope._gsQueue||(_gsScope._gsQueue=[])).push(function(){"use strict";function t(t,e,i,r){return i=parseFloat(i)-parseFloat(t),r=parseFloat(r)-parseFloat(e),Math.sqrt(i*i+r*r)}function e(t){return"string"!=typeof t&&t.nodeType||(t=_gsScope.TweenLite.selector(t),t.length&&(t=t[0])),t}function i(t,e,i){var r,s,n=t.indexOf(" ");return-1===n?(r=void 0!==i?i+"":t,s=t):(r=t.substr(0,n),s=t.substr(n+1)),r=-1!==r.indexOf("%")?parseFloat(r)/100*e:parseFloat(r),s=-1!==s.indexOf("%")?parseFloat(s)/100*e:parseFloat(s),r>s?[s,r]:[r,s]}function r(i){if(!i)return 0;i=e(i);var r,s,n,a,o,l,h,u,f=i.tagName.toLowerCase();if("path"===f)o=i.style.strokeDasharray,i.style.strokeDasharray="none",r=i.getTotalLength()||0,i.style.strokeDasharray=o;else if("rect"===f)s=i.getBBox(),r=2*(s.width+s.height);else if("circle"===f)r=2*Math.PI*parseFloat(i.getAttribute("r"));else if("line"===f)r=t(i.getAttribute("x1"),i.getAttribute("y1"),i.getAttribute("x2"),i.getAttribute("y2"));else if("polyline"===f||"polygon"===f)for(n=i.getAttribute("points").split(" "),r=0,o=n[0].split(","),"polygon"===f&&(n.push(n[0]),-1===n[0].indexOf(",")&&n.push(n[1])),l=1;n.length>l;l++)a=n[l].split(","),1===a.length&&(a[1]=n[l++]),2===a.length&&(r+=t(o[0],o[1],a[0],a[1])||0,o=a);else"ellipse"===f&&(h=parseFloat(i.getAttribute("rx")),u=parseFloat(i.getAttribute("ry")),r=Math.PI*(3*(h+u)-Math.sqrt((3*h+u)*(h+3*u))));return r||0}function s(t,i){if(!t)return[0,0];t=e(t),i=i||r(t)+1;var s=a(t),n=s.strokeDasharray||"",o=parseFloat(s.strokeDashoffset),l=n.indexOf(",");return 0>l&&(l=n.indexOf(" ")),n=0>l?i:parseFloat(n.substr(0,l))||1e-5,n>i&&(n=i),[Math.max(0,-o),n-o]}var n,a=document.defaultView?document.defaultView.getComputedStyle:function(){};n=_gsScope._gsDefine.plugin({propName:"drawSVG",API:2,version:"0.0.5",global:!0,overwriteProps:["drawSVG"],init:function(t,e){if(!t.getBBox)return!1;var n,a,o,l=r(t)+1;return this._style=t.style,e===!0||"true"===e?e="0 100%":e?-1===(e+"").indexOf(" ")&&(e="0 "+e):e="0 0",n=s(t,l),a=i(e,l,n[0]),this._length=l+10,0===n[0]&&0===a[0]?(o=Math.max(1e-5,a[1]-l),this._dash=l+o,this._offset=l-n[1]+o,this._addTween(this,"_offset",this._offset,l-a[1]+o,"drawSVG")):(this._dash=n[1]-n[0]||1e-6,this._offset=-n[0],this._addTween(this,"_dash",this._dash,a[1]-a[0]||1e-5,"drawSVG"),this._addTween(this,"_offset",this._offset,-a[0],"drawSVG")),!0},set:function(t){this._firstPT&&(this._super.setRatio.call(this,t),this._style.strokeDashoffset=this._offset,this._style.strokeDasharray=(1===t||0===t)&&.001>this._offset&&10>=this._length-this._dash?"none":this._dash+"px,"+this._length+"px")}}),n.getLength=r,n.getPosition=s}),_gsScope._gsDefine&&_gsScope._gsQueue.pop()();
