// Simple behaviour for the dashboard

var form = document.getElementById("simulate-form");
var button = document.getElementById("simulate-btn");

// When the form is submitted, show "Simulating..." on the button
form.addEventListener("submit", function () {
    button.textContent = "Simulating...";
    button.disabled = true;
});