// document.getElementById("reportLaneViolation").addEventListener("click", function () {
//     alert("Lane violation reported!");
// });

// document.getElementById("reportHighSpeedViolation").addEventListener("click", function () {
//     alert("High-speed violation reported!");
// });

// document.getElementById("reportRedLightViolation").addEventListener("click", function () {
//     alert("Red light violation reported!");
// });

// document.getElementById("reportFiningSystem").addEventListener("click", function () {
//     alert("Fining System reported!");
// });

document.getElementById("reportLaneViolation").addEventListener("click", function () {
    // Send a POST request to /start_detection when the button is clicked
    fetch('/start_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(function (response) {
        return response.json();
    })
    .then(function (data) {
        // Display the response message in an alert
        alert(data.message);
    })
    .catch(function (error) {
        console.error('Error:', error);
    });
});

document.getElementById("reportHighSpeedViolation").addEventListener("click", function () {
    // Send a POST request to /detect_high_speed when the button is clicked
    fetch('/start_detection_speed', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(function (response) {
        return response.json();
    })
    .then(function (data) {
        // Display the response message
        alert(data.message);
    })
    .catch(function (error) {
        console.error('Error:', error);
    });
});
