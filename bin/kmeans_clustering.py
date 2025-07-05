import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit,
                            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                            QComboBox, QSpinBox)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio  # 新添加的导入

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多只小鼠神经元活动分析工具")
        self.setGeometry(100, 100, 1000, 600)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 文件选择部分
        select_layout = QHBoxLayout()
        self.select_button = QPushButton("选择CSV文件")
        self.select_button.clicked.connect(self.select_files)
        select_layout.addWidget(self.select_button)
        layout.addLayout(select_layout)

        # 创建表格来显示文件和帧数范围
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['文件名', '神经元数量', '起始帧', '结束帧', '删除'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

        # K值选择部分
        k_selection_layout = QHBoxLayout()
        
        # 添加模式选择下拉框
        self.k_mode_label = QLabel("K值选择模式:")
        self.k_mode_combo = QComboBox()
        self.k_mode_combo.addItems(["自动选择", "手动设置"])
        self.k_mode_combo.currentTextChanged.connect(self.on_k_mode_changed)
        
        # 添加K值输入框
        self.k_value_label = QLabel("K值:")
        self.k_value_spin = QSpinBox()
        self.k_value_spin.setRange(2, 20)
        self.k_value_spin.setValue(3)
        self.k_value_spin.setEnabled(False)  # 默认禁用
        
        k_selection_layout.addWidget(self.k_mode_label)
        k_selection_layout.addWidget(self.k_mode_combo)
        k_selection_layout.addWidget(self.k_value_label)
        k_selection_layout.addWidget(self.k_value_spin)
        k_selection_layout.addStretch()
        
        layout.addLayout(k_selection_layout)

        # 分析按钮
        buttons_layout = QHBoxLayout()
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.analyze_data)
        buttons_layout.addWidget(self.analyze_button)
        layout.addLayout(buttons_layout)

    def on_k_mode_changed(self, text):
        self.k_value_spin.setEnabled(text == "手动设置")

    def select_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "选择CSV文件", "", "CSV files (*.csv)")
        if filenames:
            for filename in filenames:
                try:
                    df = pd.read_csv(filename)
                    num_neurons = df.shape[1] - 1
                    
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    
                    self.table.setItem(row_position, 0, QTableWidgetItem(filename))
                    self.table.setItem(row_position, 1, QTableWidgetItem(str(num_neurons)))
                    self.table.setItem(row_position, 2, QTableWidgetItem(""))
                    self.table.setItem(row_position, 3, QTableWidgetItem(""))
                    
                    delete_button = QPushButton("删除")
                    delete_button.clicked.connect(lambda _, row=row_position: self.delete_row(row))
                    self.table.setCellWidget(row_position, 4, delete_button)
                
                except Exception as e:
                    QMessageBox.warning(self, "文件读取错误", f"无法读取文件 {filename}: {str(e)}")

    def delete_row(self, row):
        self.table.removeRow(row)

    def find_optimal_k(self, X, max_k=10):
        if len(X) < max_k:
            max_k = len(X)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('肘部法则', '轮廓系数'))

        fig.add_trace(
            go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                      name='惯性'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                      name='轮廓系数'),
            row=1, col=2
        )

        fig.update_layout(
            title_text="K值选择分析",
            height=400,
            showlegend=True,
        )

        fig.write_image("k_selection_analysis.svg")
        fig.show()

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        return optimal_k

    def get_confidence_ellipse(self, points, n_std=2.0):
        """计算置信椭圆（默认使用2个标准差，约95%置信区间）"""
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        chi2_val = 2.4477 * n_std  # 95%置信区间的卡方值
        
        a = np.sqrt(chi2_val * np.abs(eigenvals[0]))
        b = np.sqrt(chi2_val * np.abs(eigenvals[1]))
        
        t = np.linspace(0, 2*np.pi, 100)
        x = mean[0] + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
        y = mean[1] + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)
        
        return x, y
    def analyze_data(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "错误", "请先选择文件")
            return

        try:
            all_neurons_data = []
            neuron_labels = []
            mouse_indices = []
            
            for row in range(self.table.rowCount()):
                filename = self.table.item(row, 0).text()
                start_frame = int(self.table.item(row, 2).text())
                end_frame = int(self.table.item(row, 3).text())
                
                df = pd.read_csv(filename)
                frame_col = df.columns[0]
                filtered_df = df[(df[frame_col] >= start_frame) & (df[frame_col] <= end_frame)]
                
                for neuron_idx in range(1, filtered_df.shape[1]):
                    neuron_data = filtered_df.iloc[:, neuron_idx].values
                    all_neurons_data.append(neuron_data)
                    neuron_labels.append(f'Mouse{row+1}_Neuron{neuron_idx}')
                    mouse_indices.append(row)

            min_length = min(len(data) for data in all_neurons_data)
            all_neurons_data = [data[:min_length] for data in all_neurons_data]
            X = np.array(all_neurons_data)

            # 在原始数据上进行聚类
            if self.k_mode_combo.currentText() == "自动选择":
                optimal_k = self.find_optimal_k(X)
            else:
                optimal_k = self.k_value_spin.value()
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(X)

            # 对数据进行标准化和PCA降维用于可视化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # 创建散点图
            fig_scatter = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # 为每个聚类添加散点和椭圆
            for i in range(optimal_k):
                mask = clusters == i
                cluster_points = X_pca[mask]
                
                # 添加散点
                fig_scatter.add_trace(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(
                        size=10,
                        color=colors[i % len(colors)],
                    ),
                    text=[f"{neuron_labels[j]}<br>Mouse {mouse_indices[j]+1}" 
                          for j in range(len(mask)) if mask[j]],
                    hovertemplate="<b>%{text}</b><br>" +
                                "PC1: %{x:.2f}<br>" +
                                "PC2: %{y:.2f}<br>" +
                                "<extra></extra>"
                ))
                
                # 添加置信椭圆 (95%)
                if len(cluster_points) > 2:
                    ellipse_x, ellipse_y = self.get_confidence_ellipse(cluster_points, n_std=2.0)
                    fig_scatter.add_trace(go.Scatter(
                        x=ellipse_x,
                        y=ellipse_y,
                        mode='lines',
                        name=f'Cluster {i} (95%)',
                        line=dict(
                            color=colors[i % len(colors)],
                            dash='dot'
                        ),
                        showlegend=True
                    ))

            # 添加解释方差比例信息
            explained_variance_ratio = pca.explained_variance_ratio_
            variance_text = (f'PC1解释方差比例: {explained_variance_ratio[0]:.2%}<br>' +
                           f'PC2解释方差比例: {explained_variance_ratio[1]:.2%}<br>' +
                           f'总解释方差比例: {sum(explained_variance_ratio):.2%}')

            fig_scatter.update_layout(
                title='神经元聚类结果 (PCA降维可视化)',
                xaxis_title='第一主成分 (PC1)',
                yaxis_title='第二主成分 (PC2)',
                width=800,
                height=600,
                showlegend=True,
                legend=dict(
                    title=dict(text='聚类'),
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            fig_scatter.add_annotation(
                text=variance_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

            fig_scatter.write_image("cluster_scatter.svg")
            fig_scatter.show()

            # 创建相关系数矩阵热图
            corr_matrix = np.corrcoef(X)

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=neuron_labels,
                y=neuron_labels,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))

            cluster_info = pd.DataFrame({
                'Neuron': neuron_labels,
                'Cluster': clusters,
                'Mouse': [f'Mouse {i+1}' for i in mouse_indices]
            })

            fig.update_layout(
                title=f'神经元活动相关性热图 (聚类数: {optimal_k})',
                xaxis_tickangle=-45,
                width=1000,
                height=1000,
            )

            fig.write_image("correlation_heatmap.svg")
            fig.show()

            cluster_stats = pd.crosstab(cluster_info['Mouse'], cluster_info['Cluster'])
            print("\n每只小鼠的神经元在各聚类中的分布：")
            print(cluster_stats)

            # 创建聚类平均活动模式热图
            cluster_means = []
            for i in range(optimal_k):
                cluster_data = X[clusters == i]
                cluster_means.append(cluster_data.mean(axis=0))

            cluster_means = np.array(cluster_means)

            fig2 = go.Figure(data=go.Heatmap(
                z=cluster_means,
                x=list(range(min_length)),
                y=[f'Cluster {i}' for i in range(optimal_k)],
                colorscale='Viridis'
            ))

            fig2.update_layout(
                title='聚类平均活动模式',
                xaxis_title='时间帧',
                yaxis_title='聚类',
                width=1000,
                height=400,
            )

            fig2.write_image("activity_heatmap.svg")
            fig2.show()

            # 创建每个cluster的曲线图
            fig3 = go.Figure()

            time_points = np.arange(min_length)
            
            for i in range(optimal_k):
                cluster_data = X[clusters == i]
                mean_curve = np.mean(cluster_data, axis=0)
                sem_curve = np.std(cluster_data, axis=0) / np.sqrt(cluster_data.shape[0])
                
                # 添加主曲线
                fig3.add_trace(go.Scatter(
                    x=time_points,
                    y=mean_curve,
                    mode='lines',
                    name=f'Cluster {i}',
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2
                    )
                ))
                
                # 添加SEM阴影
                fig3.add_trace(go.Scatter(
                    x=np.concatenate([time_points, time_points[::-1]]),
                    y=np.concatenate([mean_curve + sem_curve, (mean_curve - sem_curve)[::-1]]),
                    fill='toself',
                    fillcolor=colors[i % len(colors)],
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    opacity=0.2,
                    hoverinfo='skip'
                ))

            fig3.update_layout(
                title='聚类活动曲线 (带SEM)',
                xaxis_title='时间帧',
                yaxis_title='活动强度',
                width=1000,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                hovermode='x unified'
            )

            fig3.write_image("cluster_curves.svg")
            fig3.show()

        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
        except Exception as e:
            QMessageBox.warning(self, "处理错误", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
