# Sets up the lean alias so you can use the lean command directly
Set-Alias -Name lean -Value "C:\Users\anays\AppData\Local\Programs\Python\Python311\Scripts\lean.exe"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "               QUANTCONNECT LEAN SETUP                " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Step 1: Logging in to QuantConnect..." -ForegroundColor Yellow
Write-Host "You will be asked for your User ID and API Token."
Write-Host "Get them from: https://www.quantconnect.com/account" -ForegroundColor DarkGray
lean login

Write-Host ""
Write-Host "Step 2: Initializing Lean Project Structure..." -ForegroundColor Yellow
# Creates the lean.json workspace file in the current directory if it doesn't exist
if (-not (Test-Path "lean.json")) {
    # lean init asks for credentials again if not logged in, but login should handle it
    # We can just create a basic lean.json to mark it as a workspace
    Set-Content -Path "lean.json" -Value '{
        "environment": "backtesting",
        "cloud-id": "",
        "organization-id": ""
    }'
}

Write-Host ""
Write-Host "Step 3: Pushing project to QuantConnect Cloud..." -ForegroundColor Yellow
lean cloud push "QuantConnect_VolArb"

Write-Host ""
Write-Host "Step 4: Starting Cloud Backtest..." -ForegroundColor Green
Write-Host "The backtest will run on QuantConnect's servers using their free options data."
lean cloud backtest "QuantConnect_VolArb" --push

Write-Host ""
Write-Host "Done! Check your QuantConnect dashboard for the backtest results." -ForegroundColor Cyan
