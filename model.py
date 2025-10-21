import numpy as np
from PIL import Image
import joblib
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os

# --- 全域設定 ---
MODEL_FILE = "mnist_model.joblib"

def train_and_save_model():
    """
    獲取 MNIST 數據，訓練一個 MLPClassifier (神經網路)，並將其保存。
    """
    print("正在獲取 MNIST 數據集...")
    # 使用 'auto' 選擇最佳解析器，避免未來的警告
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target
    # 將標籤從字串轉換為整數
    y = y.astype(np.uint8)
    
    # 將像素值標準化到 0 到 1 之間
    X = X / 255.0
    
    print("正在切分訓練集與測試集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("正在訓練 MLPClassifier (神經網路)，這可能需要幾分鐘時間...")
    
    # --- 【優化後的模型設定】 ---
    # 1. 增加隱藏層的神經元數量 (256, 128)，增強模型學習能力。
    # 2. 大幅增加 max_iter 到 300，給予模型足夠的訓練時間。
    # 3. 啟用 early_stopping，如果模型在連續 10 次迭代中性能未提升，則自動停止以防過擬合。
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128), 
        max_iter=300,                  
        random_state=42,
        verbose=True,                  # 顯示訓練過程中的損失變化
        early_stopping=True,           
        n_iter_no_change=10            # 提早停止的耐心值
    )
    # --- 【優化結束】 ---

    clf.fit(X_train, y_train)
    
    print("正在計算模型準確率...")
    accuracy = clf.score(X_test, y_test)
    print(f"新模型的準確率為: {accuracy:.4f}")
    
    print(f"正在將新模型保存至 {MODEL_FILE}...")
    joblib.dump(clf, MODEL_FILE)
    print("模型已成功保存。")

def load_model():
    """
    從檔案中載入已訓練的模型。
    如果模型檔案不存在，則會自動訓練並保存一個新模型。
    """
    if not os.path.exists(MODEL_FILE):
        print("模型檔案不存在，正在訓練一個新模型...")
        train_and_save_model()
        
    print(f"正在從 {MODEL_FILE} 載入模型...")
    model = joblib.load(MODEL_FILE)
    print("模型載入成功。")
    return model

def preprocess_image(image_file):
    """
    對上傳的圖片進行預處理，使其符合 MNIST 模型的輸入要求。
    - 開啟圖片
    - 轉換為灰階 ('L')
    - 縮放至 28x28 像素
    - 轉換為 numpy 陣列
    - 反轉顏色 (MNIST 是白字黑底)
    - 標準化像素值
    - 將 28x28 的圖片攤平成 784 維的一維陣列
    """
    try:
        img = Image.open(image_file).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # 檢查是否需要反轉顏色。標準 MNIST 是白字黑底。
        # 大多數用戶繪製的圖是黑字白底。
        # 我們透過比較角落和中心的平均像素值來判斷。
        corners_mean = np.mean(img_array[:5, :5]) + np.mean(img_array[-5:, -5:])
        center_mean = np.mean(img_array[10:18, 10:18])
        
        # 如果角落比中心亮，代表是黑字白底，需要反轉。
        if corners_mean > center_mean:
            img_array = 255 - img_array

        # 標準化
        img_array = img_array / 255.0
        
        # 攤平成一維陣列
        img_flat = img_array.flatten()
        
        # 重塑為 (1, 784) 以便進行單一樣本預測
        return img_flat.reshape(1, -1) 
    except Exception as e:
        print(f"圖片預處理時發生錯誤: {e}")
        return None

def predict(model, processed_image):
    """
    對一個已經預處理過的圖片進行預測。
    返回預測的數字和信賴度分數。
    """
    if processed_image is None:
        return None, None
        
    try:
        probabilities = model.predict_proba(processed_image)
        prediction = model.classes_[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        return str(prediction), confidence
    except Exception as e:
        print(f"模型預測時發生錯誤: {e}")
        return None, None

# --- 主程式進入點 ---
# 當這個腳本被直接執行時，會觸發模型訓練。
if __name__ == '__main__':
    train_and_save_model()