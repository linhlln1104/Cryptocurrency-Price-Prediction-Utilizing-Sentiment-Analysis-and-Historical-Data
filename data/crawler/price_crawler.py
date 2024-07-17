from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    options = Options()
    options.headless = False  # Set to True for production, False for debugging
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def move_to_data_point_and_extract(driver, canvas, x, y):
    action = ActionChains(driver)
    action.move_to_element_with_offset(canvas, x, y).perform()
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'chart-title-indicator-container')))  # Ensure you know the tooltip class
    tooltip = driver.find_element(By.CLASS_NAME, 'chart-title-indicator-container')
    return tooltip.text

if __name__ == "__main__":
    driver = setup_driver()
    try:
        driver.get('https://www.binance.com/vi/trade/BTC_USDT?_from=markets&type=spot')
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'canvas')))
        canvas = driver.find_element(By.TAG_NAME, 'canvas')

        # Example pixel coordinates for demonstration (you need to adjust these based on your actual data layout)
        data_points = [(150, 300), (200, 300), (250, 300)]  # (x, y) tuples

        for x, y in data_points:
            data = move_to_data_point_and_extract(driver, canvas, x, y)
            print(f"Data at ({x}, {y}): {data}")
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        driver.quit()
