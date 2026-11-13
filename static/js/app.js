
document.addEventListener("DOMContentLoaded", function () {

    // THEME TOGGLE
    const themeToggle = document.getElementById("themeToggle");

    // Load saved theme
    const savedTheme = localStorage.getItem("smarttrip-theme");

    if (savedTheme === "dark") {
        document.body.classList.add("dark-theme");
    } else {
        document.body.classList.remove("dark-theme");
    }

    // Update button text
    function updateThemeButton() {
        if (themeToggle) {
            themeToggle.textContent =
                document.body.classList.contains("dark-theme")
                    ? "Switch to Light Mode"
                    : "Switch to Dark Mode";
        }
    }

    updateThemeButton();

    // Theme button click
    if (themeToggle) {
        themeToggle.addEventListener("click", function () {

            document.body.classList.toggle("dark-theme");

            const currentTheme =
                document.body.classList.contains("dark-theme")
                    ? "dark"
                    : "light";

            localStorage.setItem("smarttrip-theme", currentTheme);

            updateThemeButton();
        });
    }

    // FORM SUBMISSION PROTECTION
    const forms = document.querySelectorAll("form");

    forms.forEach(function (form) {
        form.addEventListener("submit", function () {
            const submitButton =
                form.querySelector('button[type="submit"]');

            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = "Please wait...";
            }
        });
    });

    // CONFIRM DELETE ACTIONS
    const deleteForms =
        document.querySelectorAll(".delete-form");

    deleteForms.forEach(function (form) {
        form.addEventListener("submit", function (event) {
            const confirmed =
                confirm("Are you sure you want to delete this item?");

            if (!confirmed) {
                event.preventDefault();
            }
        });
    });

    // AUTO HIDE FLASH MESSAGES
    const flashMessages =
        document.querySelectorAll(".flash-message");

    flashMessages.forEach(function (message) {
        setTimeout(function () {
            message.style.opacity = "0";

            setTimeout(function () {
                message.remove();
            }, 400);

        }, 4000);
    });

});
