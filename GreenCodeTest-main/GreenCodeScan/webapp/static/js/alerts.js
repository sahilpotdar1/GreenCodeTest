// Alert system module
export function showAlert(message, isSuccess = false) {
    const alertBox = document.getElementById('alert-box');
    alertBox.textContent = message;
    alertBox.className = 'alert ' + (isSuccess ? 'alert-success' : 'alert-error');
    alertBox.style.display = 'block';

    // Auto-hide alert after 5 seconds
    setTimeout(() => {
        hideAlert(alertBox);
    }, 5000);
}

function hideAlert(alertBox) {
    alertBox.style.display = 'none';
}