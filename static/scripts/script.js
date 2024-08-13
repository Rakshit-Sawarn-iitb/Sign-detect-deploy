document.addEventListener('DOMContentLoaded', function() {
    fetch('/check-login-status', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
        credentials: 'include'
    })
    .then(response => response.json())
    .then(data => {
        const loginButton = document.getElementById('login-button');
        const logoutButton = document.getElementById('logout-button');

        if (data.loggedIn) {
            // User is logged in
            loginButton.style.display = 'none'; // Hide login button
            logoutButton.style.display = 'block'; // Show logout button
            
            logoutButton.addEventListener('click', async function() {
                const response = await fetch('/logout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'include'
                });

                const result = await response.json();
                alert(result.message);

                if (response.ok) {
                    // Redirect to home page after successful logout
                    window.location.href = '/';
                }
            });
        } else {
            // User is not logged in
            loginButton.style.display = 'block'; // Show login button
            logoutButton.style.display = 'none'; // Hide logout button
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        // Handle fetch error
        const loginButton = document.getElementById('login-button');
        const logoutButton = document.getElementById('logout-button');
        loginButton.style.display = 'none'; // Hide login button on error
        logoutButton.style.display = 'none'; // Hide logout button on error
    });
});


document.getElementById('signatureForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    try {
        const response = await fetch('/compare', {
            method: 'POST',
            body: formData,
            credentials: 'include' // Include cookies (if using session-based authentication)
        });

        const result = await response.json();
        alert(result.is_genuine || result.error);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while verifying the signature.');
    }
});