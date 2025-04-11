# **TikTok Claims Classification: End-to-End Analysis and Modeling**  

This project explores how TikTok video features—such as engagement metrics and author verification status—influence the classification of content as claims or opinions using machine learning in Python. By analyzing relationships between views, likes, shares, and account status (e.g., verified or banned), I built predictive models to automate content moderation and prioritize high-risk claims for review. The project leverages Python libraries like pandas, scikit-learn, XGBoost, and Tableau for data analysis, statistical testing, and visualization.

### **Project Navigation: Two-Part Structure**

To ensure clarity, organization, and optimal performance on GitHub, I have divided the project into **two logical parts**:

- **[Part 1 – From Project Proposal to EDA](https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok_Claims_Classification_Part1_Project_Proposal_to_EDA.ipynb):** This notebook establishes the foundation of the project through stakeholder planning, exploratory data analysis (EDA), and early insights into TikTok engagement and content patterns.

https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok_Claims_Classification_Part1_Project_Proposal_to_EDA.ipynb

- **[Part 2 – From Statistical Testing to Machine Learning](https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok_Claims_Classification_Part2_Statistical_Analysis_to_Modeling.ipynb):** This notebook transitions from analysis to modeling. It includes hypothesis testing, multiple logistic regression, and machine learning models such as Random Forest and XGBoost for automated claims classification.

https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok_Claims_Classification_Part2_Statistical_Analysis_to_Modeling.ipynb

Each part is structured around key phases, making it easier to follow the data journey from exploration to model deployment. You can explore both parts in sequence using the links provided above, or jump directly to a specific phase of interest.

## **Project Overview**  

The **TikTok Claims Classification** project aims to:  

- **Analyze Engagement Differences**: Compare view counts, likes, and shares between claim and opinion videos  
- **Evaluate Verification Impact**: Assess how verified/unverified status affects content classification  
- **Build Predictive Models**: Develop machine learning classifiers to automatically detect claims  
- **Optimize Moderation**: Reduce manual review workload by prioritizing likely claims  
- **Ensure Ethical Implementation**: Minimize false negatives while maintaining fairness  

## **Dataset Structure**  

The dataset contains **19,383 TikTok videos**, each representing a unique published video labeled as either a claim or opinion. Key features include:

- **claim_status**: Target variable indicating whether the video contains a “claim” (unsourced/unverified info) or an “opinion” (personal belief)  
- **video_id**: Unique identifier assigned to each video upon publication  
- **video_duration_sec**: Duration of the video in seconds (continuous)  
- **video_transcription_text**: Transcribed spoken content from the video (natural language)  
- **verified_status**: Indicates if the video author is verified (verified / not verified)  
- **author_ban_status**: Indicates the author's moderation status (active / under review / banned)  
- **video_view_count**: Total number of views (continuous)  
- **video_like_count**: Total number of likes (continuous)  
- **video_share_count**: Total number of shares (continuous)  
- **video_download_count**: Total number of downloads (continuous)  
- **video_comment_count**: Total number of comments (continuous)  

This dataset serves as the foundation for analyzing patterns in TikTok content moderation and building a classification model to distinguish between claims and opinions.

## **Data Processing and Analysis Steps**  

### **Data Cleaning**  
- Removed 298 rows with missing values (1.5% of dataset)  
- Capped extreme outliers in engagement metrics using IQR method  
- Balanced classes via upsampling (50.3% claims vs 49.7% opinions)  

### **Exploratory Data Analysis**  
- Visualized engagement distributions: Claims showed 2.5× higher median views  
- Identified banned authors generate 3× more claims than active accounts  
- Created Tableau dashboards comparing claim/opinion metrics  

### **Statistical Testing**  
- Conducted t-tests: Verified vs unverified accounts show significant engagement differences (p<0.001)  
- Confirmed claims drive higher shares (mean=16,735) vs opinions (mean=14,111)  

### **Machine Learning Modeling**  
- **Feature Engineering**:  
  - Extracted text length from transcriptions  
  - Generated n-gram features via CountVectorizer  
  - One-hot encoded categorical variables  
- **Model Development**:  
  - Logistic Regression baseline (67% accuracy)  
  - Random Forest (99.5% recall)  
  - XGBoost (99.0% recall)  
- **Evaluation**:  
  - Prioritized recall to minimize false negatives  
  - Final RF model achieved 99.48% recall, 99.95% precision  

## **Key Insights**  

### **Statistical Findings**  
- **Claims generate significantly higher engagement**:  
  - 2.1× more views (p<0.001)  
  - 1.8× more likes (p<0.001)  
- **Banned accounts are 3× more likely to post claims**  
- **Verified accounts post fewer claims** (32% vs 68% for unverified)  

### **Model Performance**

| Model               | Recall  | Precision | F1-Score |
|---------------------|---------|-----------|----------|
| Logistic Regression | 82%     | 63%       | 71%      |
| Random Forest       | 99.48%  | 99.95%    | 99.71%   |
| XGBoost             | 98.98%  | 99.90%    | 99.44%   |


### **Feature Importance**  
1. **Video share count** (28% importance)  
2. **Author ban status** (22%)  
3. **Video view count** (19%)  
4. **Text length** (15%)  

## **Project Highlights**  

- Developed **end-to-end classification pipeline** from EDA to deployment-ready model  
- Achieved **near-perfect recall (99.48%)** to minimize harmful content oversight  
- Identified **key moderation signals**: High shares + banned author status  
- Created **interpretable visualizations** of engagement patterns  
- Established **ethical framework** for automated content moderation  

## **Future Work**  

- **Incorporate Advanced NLP**: Implement BERT for semantic claim detection  
- **Expand Feature Set**: Add user history and temporal engagement patterns  
- **Real-Time Testing**: Shadow deployment to assess production performance  
- **Fairness Audits**: Evaluate demographic bias in classification  
- **Multi-Language Support**: Adapt model for non-English content  

This analysis provides **scalable solutions for content moderation**, enabling TikTok to efficiently identify high-risk claims while reducing manual review workload. The findings demonstrate how engagement metrics and account characteristics can reliably predict problematic content.
