# Angels Dataset 長者照護分析項目
# Elder Care Analysis Project - Angels Dataset

## 📋 項目概述 Project Overview

本項目基於 Angels Dataset（420,510 筆長者資料）進行全面的探索性數據分析和長照服務使用者輪廓分析，旨在為台灣長照政策提供實證基礎。

This project conducts comprehensive exploratory data analysis and long-term care user profiling based on the Angels Dataset (420,510 elderly records) to provide evidence-based insights for Taiwan's long-term care policy.

## 🗂️ 項目結構 Project Structure

```
📁 Angels-Analysis/
├── 📁 analysis/           # 分析腳本 Analysis Scripts
│   ├── eda_angels.py                    # 探索性數據分析
│   ├── vulnerability_indices.py         # 脆弱性指數分析  
│   └── long_term_care_profile_analysis.py # 長照使用者輪廓分析
├── 📁 data/              # 數據文件 Data Files
│   └── dataset/angels.csv               # 原始數據集
├── 📁 results/           # 分析結果 Analysis Results
│   ├── EDA_Summary_Report.md            # EDA總結報告
│   ├── Vulnerability_Indices_Report.md  # 脆弱性指數報告
│   ├── Long_Term_Care_Comprehensive_Report.md # 長照綜合分析報告
│   └── *.csv                           # 統計結果數據
├── 📁 visualizations/    # 視覺化圖表 Visualizations
│   ├── pairplot.png                    # 變數關係圖
│   ├── vulnerability_indices_analysis.png # 脆弱性指數分析圖
│   ├── long_term_care_profile_analysis.png # 長照輪廓分析圖
│   └── *.png                           # 其他圖表
├── 📁 docs/              # 文檔資料 Documentation
└── 📁 venv/              # 虛擬環境 Virtual Environment
```

## 🎯 分析內容 Analysis Content

### 1. 📊 探索性數據分析 (EDA)
- **文件**: `analysis/eda_angels.py`
- **報告**: `results/EDA_Summary_Report.md`
- **圖表**: `visualizations/pairplot.png` 等

**主要發現**:
- 420,510 筆長者記錄，年齡範圍 65-118 歲
- 長照使用率僅 2.88%，高度不平衡
- 年齡是最強預測因子

### 2. 🏥 脆弱性指數分析
- **文件**: `analysis/vulnerability_indices.py`
- **報告**: `results/Vulnerability_Indices_Report.md`
- **圖表**: `visualizations/vulnerability_indices_analysis.png`

**四大指數**:
- 🏠 **孤獨指數**: 37.54% (獨居+老老照顧比例)
- 🏥 **生理脆弱性指數**: 4.00% (嚴重身心障礙比例)
- 🏠 **居住障礙指數**: 13.19% (無電梯公寓2樓以上)
- 🚌 **生活機能不便指數**: 1.03 (交通+購物+醫療便利性)

### 3. 👨‍👩‍👧‍👦 長照使用者輪廓分析
- **文件**: `analysis/long_term_care_profile_analysis.py`
- **報告**: `results/Long_Term_Care_Comprehensive_Report.md`
- **圖表**: `visualizations/long_term_care_profile_analysis.png`

**核心發現**:
- **年齡效應**: 長照使用者平均 81.1 歲 vs 非使用者 73.4 歲
- **家庭悖論**: 獨居者使用率僅比同住者高 0.37%
- **經濟主導**: 有補助者使用率 13.86% vs 無補助者 2.71%

### 4. 🔍 年齡與子女數關係驗證
- **文件**: `analysis/age_children_relationship_validation.py`
- **報告**: `results/Age_Children_Validation_Report.md`
- **圖表**: `visualizations/age_children_relationship_validation.png`

**驗證發現**:
- **混淆效應確認**: 年齡與子女數相關性 r=0.329 (中等相關)
- **分層分析結果**: 71.4% 年齡組仍顯示子女數顯著效應
- **年齡調節效應**: 子女數影響隨年齡增加而減弱
- **U型關係確認**: 2-3個子女時長照使用率最低

### 5. 🏠 高齡資產盤點分析
- **文件**: `analysis/elderly_housing_asset_analysis.py`
- **報告**: `results/Elderly_Housing_Asset_Report.md`
- **圖表**: `visualizations/elderly_housing_asset_analysis.png`

**核心發現**:
- **房產老化嚴重**: 46.9% 高齡屋主持有30年以上老屋
- **以房養老困境**: 僅39.0% 房產適合以房養老
- **地理高度集中**: 前20大地區集中37.5%的高齡房產
- **繼承壓力巨大**: 36個熱區面臨大量房產釋出壓力

### 6. 🌍 環境因素對健康影響分析
- **文件**: `analysis/environmental_health_impact_analysis.py`
- **報告**: `results/Environmental_Health_Impact_Report.md`
- **圖表**: `visualizations/environmental_health_impact_analysis.png`

**意外發現**:
- **環境因素無顯著影響**: 生活機能、居住障礙、家庭支持均不顯著
- **年齡主導健康**: 年齡是唯一重要的健康決定因子
- **刻板印象挑戰**: 老老照顧長者健康狀況最佳
- **健康排序**: 老老照顧 > 獨居 > 與家人同住

## 🚀 快速開始 Quick Start

### 環境設置
```bash
# 1. 創建虛擬環境
python3 -m venv venv

# 2. 激活虛擬環境
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 運行分析
```bash
# 1. 探索性數據分析
python analysis/eda_angels.py

# 2. 脆弱性指數分析
python analysis/vulnerability_indices.py

# 3. 長照使用者輪廓分析
python analysis/long_term_care_profile_analysis.py
```

## 📈 主要發現 Key Findings

### 🎯 長照使用者特徵
- **使用率**: 2.88% (12,100/420,510)
- **平均年齡**: 81.1 歲
- **年齡分布**: 85歲後需求急遽上升
- **障礙程度**: 重度障礙者使用率最高 (21.73%)

### 👨‍👩‍👧‍👦 家庭因素影響
- **獨居 vs 同住**: 差異僅 0.37% (3.21% vs 2.84%)
- **子女數量**: 呈現U型關係，2個子女時使用率最低 (經年齡驗證確認)
- **年齡調節效應**: 子女數影響在65-84歲明顯，85歲後減弱
- **地理距離**: 影響有限 (同縣市 2.92% vs 異縣市 2.60%)

### 💰 經濟因素主導
- **福利效應**: 有補助 13.86% vs 無補助 2.71%
- **財富效應**: 有殼階級使用率反而較低 (1.82% vs 3.35%)
- **收入分層**: 呈現L型關係，非U型

### 🏥 脆弱性地圖
- **高風險區域**: 綜合脆弱性分數 17.91
- **關鍵指標**: 孤獨指數變異最大，是政策重點
- **居住障礙**: 13%長者面臨無電梯困擾

### 🏠 高齡資產現況
- **房產老化**: 46.9% 持有30年以上老屋，28.1% 持有40年以上
- **以房養老適合率**: 僅39.0%，屋齡過高是主要障礙
- **地理集中**: F14、F18、F05等地區為繼承壓力熱區
- **市場潛力**: 放寬條件可提升適合率至55-60%

### 🌍 環境健康關係
- **意外結果**: 所有環境因素對健康均無統計顯著影響
- **年齡主導**: 年齡是唯一重要的健康決定因子 (勝算比1.035)
- **家庭支持**: 老老照顧(2.024) > 獨居(2.025) > 與家人同住(2.031)
- **政策啟示**: 環境改善效果可能被高估，需重新檢視政策優先級

## 📊 統計檢定結果

| 檢定項目 | 卡方統計量 | p值 | 顯著性 |
|----------|------------|-----|--------|
| 家庭型態影響 | 75.95 | < 0.001 | ✅ |
| 子女數量影響 | 3,974.93 | < 0.001 | ✅ |
| 房屋類型影響 | 1,027.05 | < 0.001 | ✅ |
| 收入類型影響 | 3,006.02 | < 0.001 | ✅ |

## 🎯 政策建議 Policy Recommendations

### 短期措施 (1年內)
1. **擴大補助範圍**: 提高補助門檻至中低收入戶1.5倍
2. **簡化申請流程**: 建立一站式服務窗口
3. **分層服務策略**: 針對不同年齡組設計差異化服務

### 中期措施 (2-3年)
1. **分級付費制度**: 根據收入和家庭結構設計差異化自付額
2. **家庭支持服務**: 特別關注無子女和獨子女家庭
3. **專業人力培訓**: 擴大照護員培訓，減少家庭依賴

### 長期措施 (5年以上)
1. **年齡分層政策**: 85歲以上以專業照護為主，65-84歲強化家庭支持
2. **整合照護體系**: 建立連續性照護，考慮年齡調節效應
3. **科技輔助照護**: AI和IoT提升效率，補強家庭照護不足

## 🔧 技術規格 Technical Specifications

### 數據規模
- **記錄數**: 420,510 筆
- **變數數**: 15 個
- **數據大小**: 48.12 MB
- **年齡範圍**: 65-118 歲

### 分析工具
- **Python**: 3.9+
- **主要套件**: pandas, numpy, matplotlib, seaborn, scipy
- **統計方法**: 卡方檢定、描述統計、交叉分析
- **視覺化**: 直方圖、散點圖、熱力圖、箱型圖

### 系統需求
- **記憶體**: 建議 8GB 以上
- **儲存空間**: 1GB 以上
- **作業系統**: macOS, Linux, Windows

## 📚 相關文獻 References

1. 內政部統計處 (2024). 人口統計年報
2. 衛生福利部 (2024). 長期照顧十年計畫2.0
3. 國家發展委員會 (2024). 人口推估報告

## 👥 貢獻者 Contributors

- **分析師**: Kiro AI Assistant
- **數據來源**: Angels Dataset
- **技術支持**: Python Data Science Stack

## 📄 授權 License

本項目僅供學術研究使用，請勿用於商業用途。

---

*最後更新: 2025年1月18日*  
*項目版本: v1.0*  
*聯絡方式: 請透過 GitHub Issues 聯繫*