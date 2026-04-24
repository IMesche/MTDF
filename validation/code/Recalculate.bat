@echo off
REM Regenerate MTDF Validation Dashboard from Workbook
REM Run this after modifying parameters, targets, or uncertainties in the workbook

echo Regenerating MTDF Validation Dashboard...
echo.

python run_validate.py --workbook ../data/DB_Workbook_STRICT_V17.xlsx --out UI/Validation_Dashboard_V17.html

echo.
echo Done! Refresh the dashboard in your browser (F5) to see changes.
pause
