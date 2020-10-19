var htmlOriginal;
var htmlMem;

//////////////////////////////////////////////////////////////////////////////////
function onSplitFrames() {
    // -- called when splitting banner into frames
}

// INIT BANNER ----------------------------------
function init() {
    // Disable separate frames
    // separateFrames = function() {};

    setBannerSize();
    show(".scene");
    // show(".common");
    set(".htmlMem", { skewX: 0.1, transformOrigin: "50% 50%" });

    new ImagePreloader(
        ["frame1-text.png", "frame2-text.png", "frame3-text.png", 'background.jpg', 'background-gradient.jpg'],
        frame0
    );

    htmlOriginal = $("body").innerHTML;
    htmlMem = $(".htmlMem").innerHTML;

    addListeners();
}

function frame0() {
    show(".banner");
    from(".htmlMem", 0.25, { opacity: 0 }, "in", 0);
    frame1();
}

function frame1() {
    show(".frame1");
    from(".frame1", 0.4, { opacity: 0, y: -20 }, "out", 0.1);

    wait(2.0, frame2);
    to(".background", 2.5, { y: -60}, "in")
}

function frame2() {
    show(".frame2");
    from(".frame2 .text", 0.4, { opacity: 0, y: -20 }, "out", 0.5);
    to(".frame1", 0.4, { opacity: 0, y: 20 }, "in");
    to(".background", 0.5, { opacity: 0, transformOrigin: "50% 90%", scale: 0.7}, "in")

    set('.frame2 .cta', { display: "none" })

    var timeline = new TimelineMax({ delay: 0.5 });
    timeline.set('.frame2 .cta', { display: "" }, 0.01)
        .from('.frame2 .cta', 0.3, { scale: 0.8, ease: Back.easeOut })
        .from('.frame2 .cta', 0.3, { x: 101, width: 38, ease: Power2.easeOut }, 0)
        .from('.frame2 .cta img', 0.2, { opacity: 0, ease: Power2.easeIn }, 0.1)

    wait(1.0, frame3);
}

function frame3() {
    show(".frame3");

    var timeline = new TimelineMax({ delay: 0 });
    
    // cursor slides in
    timeline.from('.cursor', 0.5, { x: 300, ease: Power2.easeOut})
        .from('.cursor', 0.7, { rotation: -90, ease: Back.easeOut}, 0.2)
        .from('.cursor .circle', 0.3, { opacity: 0.6, scale: 0, transformOrigin: "50% 50%", ease: Power2.easeOut}, 1.2)
        .to('.cursor', 0.2, { scale: 0.7, transformOrigin: "50% 50%", ease: Power3.easeOut}, 1.2)
        .to('.cursor', 0.2, { scale: 1, transformOrigin: "50% 50%", ease: Power3.easeOut}, 1.3)

        // button press
        .to('.frame2 .cta', 0.15, { boxShadow: "0 0px 0px rgba(0,0,0,0.5)", scale: 0.7, ease: Power2.easeOut }, 1.2)
        .to(".frame2 .cta .logo-blue", 0.2, { opacity: 0, ease: Power2.easeOut }, "-=0.15")
        .to('.frame2 .text', 0.5, { opacity: 0, transformOrigin: "50% 60%", scale: 0.7, ease: Power2.easeIn }, "-=0.2")
        .to('.frame2 .cta', 0.3, { boxShadow: "0 3px 7px rgba(0,0,0,0.3)", backgroundColor: "#009cde", scale: 1, ease: Power2.easeOut }, "-=0.3")
        .to(".frame2 .cta .text-white", 0.3, { opacity: 1, ease: Power2.easeOut }, "-=0.3")
        .from(".frame3 .text", 0.4, { opacity: 0, ease: Power2.easeIn }, "+=0")
        
        // cursor slides out
        .to('.cursor', 0.6, { y: 150, rotation: 90, ease: Power2.easeIn}, 1.3)
        .to('.cursor', 0.6, { x: 150, ease: Sine.easeIn}, 1.3)

        // logo CTA up
        .from(".frame3 .logo", 0.5, { y: 15, opacity: 0, ease: Power2.easeOut }, "-=0.4")


    // .from('.frame3 .cta img', 0.25, { opacity: 0, ease: Power2.easeIn }, "-=0.25")


    if (banner.loopCount < 1) {
        banner.loopCount++;
        wait(5, loop);
    } else {
        from(".replay", 0.5, { opacity: 0 }, "in", 1);
    }
}

function loop() {
    killAll();
    to(".htmlMem", 1.0, { opacity: 0 }, "in");
    set(".replay", { display: "none", opacity: 1 });

    banner.looping = true;

    wait(1.0, function () {
        killAll();
        set(".htmlMem", { opacity: 1 });
        $(".htmlMem").innerHTML = htmlMem;
        frame0();
        banner.looping = false;
    });
}

function reset() {
    loop();
}

window.addEventListener("load", init);
