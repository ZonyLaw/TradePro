document.addEventListener("DOMContentLoaded", function () {
  var alertClose = document.querySelectorAll(".alert__close");

  // Loop through each close button
  for (var i = 0; i < alertClose.length; i++) {
    alertClose[i].addEventListener("click", function () {
      // Get the parent element of the button
      var alertContainer = this.parentElement;
      alertContainer.style.display = "none";
    });
  }
});
