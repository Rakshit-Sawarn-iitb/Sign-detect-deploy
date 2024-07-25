var typed = new Typed(".auto-type", {
    strings: ["Verify"],
    typeSpeed: 200,
    backSpeed: 250,
    loop: true
})

document.getElementById('signatureForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file1', document.getElementById('file1').files[0]);
    formData.append('file2', document.getElementById('file2').files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').textContent = `Error: ${data.error}`;
        } else {
            document.getElementById('result').textContent = `Result: ${data.is_genuine}`;
            console.log(data.is_genuine);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = `Error: ${error}`;
    });
});