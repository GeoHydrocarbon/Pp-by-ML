import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pandas as pd

'''
输入文件的要求：

    1. 文件格式要求
    ​文件类型​：CSV格式（逗号分隔值文件）
    ​编码​：支持多种编码（utf-8, gbk, gb2312, gb18030, latin1），程序会自动尝试解码
    2. 数据列要求
    ​必须包含深度列​：数据中必须包含以下任一名称的深度列（不区分大小写）：
    Depth, DEPTH, 深度, MD, TVD
    ​测井曲线列​：其他列应为数值型测井曲线数据，常见的曲线名称包括（但不限于）：
    AC (声波时差)
    CN (中子孔隙度)
    GR (自然伽马)
    DEN (密度)
    LLD (深电阻率)
    LLS (浅电阻率)
    POR (孔隙度)
    CAL (井径)
    SP (自然电位)
    Sw (含水饱和度)
    SH (泥质含量)
    3. 数据类型要求
    ​深度列​：应为数值型数据（浮点数或整数）
    ​测井曲线列​：应为数值型数据（浮点数或整数），非数值型列会被跳过
    4. 数据内容要求
    ​有效范围​：程序内置了常见测井曲线的有效范围，超出范围的值会被过滤：
'''
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class WellLogProcessor:
    def __init__(self, df):
        self.df = df
        self.processed_df = df.copy()
        self.filtered_data = {}  # 存储每条曲线过滤后的数据
        
        # 查找深度列
        depth_columns = ['Depth', 'DEPTH', '深度', 'MD', 'TVD']
        self.depth_col = None
        for col in depth_columns:
            if col in df.columns:
                self.depth_col = col
                break
                
        if self.depth_col is None:
            raise ValueError("未找到深度列，请确保数据中包含Depth、DEPTH、深度、MD或TVD列")
            
        # 重命名深度列为'Depth'
        self.df = self.df.rename(columns={self.depth_col: 'Depth'})
        self.processed_df = self.processed_df.rename(columns={self.depth_col: 'Depth'})
        
        # 打印所有列名
        print("\n数据文件中的列名：")
        for col in self.df.columns:
            print(f"- {col}")
        
    def filter_anomalies(self):
        """过滤异常值"""
        # 定义各测井曲线的有效范围
        log_ranges = {
            'AC': (0, 200),      # us/ft
            'CN': (0, 100),      # 假设范围
            'GR': (0, 150),      # API
            'DEN': (1.5, 3.0),   # g/cm3
            'LLD': (0, 100),     # ohm.m
            'LLS': (0, 100),     # ohm.m
            'POR': (0, 1),       # 孔隙度
            'PORF': (0, 1),      # 有效孔隙度
            'VPOR': (0, 1),      # 孔隙度
            'CAL': (0, 40),      # 井径cm
            'SP': (0, 100),      # 自然电位
            'Sw': (0, 1),        # 含水饱和度
            'SH': (0, 100),
        }
        
        # 对每条曲线进行过滤
        for column in self.df.columns:
            if column == 'Depth':
                continue
                
            # 只处理数值型列
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                print(f"跳过非数值型列: {column}")
                continue
                
            # 获取当前曲线的有效范围
            if column in log_ranges:
                min_val, max_val = log_ranges[column]
                
                # 特殊处理AC的单位转换
                if column == 'AC':
                    if self.df[column].mean() > 1000:  # 如果平均值大于1000，可能是us/m
                        self.processed_df[column] = self.df[column] / 3.28084  # 转换为us/ft
                
                # 创建掩码
                mask = (self.processed_df[column] >= min_val) & (self.processed_df[column] <= max_val)
                self.filtered_data[column] = self.processed_df[mask].copy()
                print(f"处理曲线 {column}: 原始点数 {len(self.df)}, 过滤后点数 {len(self.filtered_data[column])}")
            else:
                # 对于没有定义范围的曲线，直接使用原始数据
                self.filtered_data[column] = self.processed_df.copy()
                print(f"处理曲线 {column}: 未定义有效范围，使用原始数据")
                
        return self.processed_df
    
    def merge_filtered_data(self):
        """合并所有过滤后的数据，只保留所有曲线都有值的深度点"""
        # 从保存的CSV文件中读取数据
        filtered_dfs = {}
        for column in self.filtered_data.keys():
            filename = f'filtered_{column}.csv'
            try:
                filtered_dfs[column] = pd.read_csv(filename)
                print(f"从 {filename} 读取到 {len(filtered_dfs[column])} 个点")
            except FileNotFoundError:
                print(f"警告: 未找到文件 {filename}")
                continue
        
        if not filtered_dfs:
            return None
        
        # 获取所有深度点
        common_depths = None
        for column, df in filtered_dfs.items():
            current_depths = set(df['Depth'].values)
            print(f"曲线 {column} 的深度点数量: {len(current_depths)}")
            if common_depths is None:
                common_depths = current_depths
            else:
                common_depths = common_depths.intersection(current_depths)
            print(f"当前共同深度点数量: {len(common_depths)}")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({'Depth': sorted(common_depths)})
        
        # 合并每条曲线的数据
        for column, df in filtered_dfs.items():
            result_df = result_df.merge(
                df[['Depth', column]], 
                on='Depth', 
                how='inner'
            )
        
        return result_df

class InteractiveScatterEditor:
    def __init__(self, x, y, title, a=-0.0001995, b=5.0099):
        self.x = x
        self.y = y
        self.original_x = x.copy()
        self.original_y = y.copy()
        self.title = title
        self.a = a
        self.b = b
        self.should_continue = True  # 添加控制变量
        
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # 绘制初始散点图
        self.scatter = self.ax.scatter(self.x, self.y, s=10, alpha=0.5)
        
        # 绘制趋势线
        if title == 'AC':
            depth_range = np.linspace(min(self.y), max(self.y), 100)
            trend_line = np.exp(self.a * depth_range + self.b)
            self.trend_line, = self.ax.plot(trend_line, depth_range, 'r-', label='正常AC趋势线')
            self.ax.legend()
        
        self.ax.set_title(f'{title} - 点击拖动框选要删除的区域，右键点击重置')
        
        # 根据实际深度范围设置y轴范围
        depth_min = np.floor(min(self.y) / 100) * 100
        depth_max = np.ceil(max(self.y) / 100) * 100
        self.ax.set_ylim(depth_max, depth_min)
        
        # 添加矩形选择器
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
        )
        
        # 添加按钮 - 重新排布到一行
        button_width = 0.2
        button_spacing = 0.05
        start_x = 0.1
        
        ax_prev = plt.axes([start_x, 0.05, button_width, 0.075])
        ax_next = plt.axes([start_x + button_width + button_spacing, 0.05, button_width, 0.075])
        ax_save = plt.axes([start_x + 2*(button_width + button_spacing), 0.05, button_width, 0.075])
        ax_reset = plt.axes([start_x + 3*(button_width + button_spacing), 0.05, button_width, 0.075])
        
        self.btn_prev = plt.Button(ax_prev, '上一条曲线')
        self.btn_next = plt.Button(ax_next, '下一条曲线')
        self.btn_save = plt.Button(ax_save, '保存剩余点')
        self.btn_reset = plt.Button(ax_reset, '重置所有点')
        
        self.btn_prev.on_clicked(self.previous_curve)
        self.btn_next.on_clicked(self.next_curve)
        self.btn_save.on_clicked(self.save_points)
        self.btn_reset.on_clicked(self.reset_points)
        
        # 右键点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_right_click)
        
    def on_select(self, eclick, erelease):
        """当矩形选择完成时调用"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # 确保x1 < x2和y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 创建选择掩码
        mask = ~((self.x > x1) & (self.x < x2) & (self.y > y1) & (self.y < y2))
        
        # 更新数据
        self.x = self.x[mask]
        self.y = self.y[mask]
        
        # 更新散点图
        self.update_plot()
    
    def on_right_click(self, event):
        """右键点击重置视图"""
        if event.button == 3:  # 右键
            self.reset_points(None)
    
    def reset_points(self, event):
        """重置所有点"""
        self.x = self.original_x.copy()
        self.y = self.original_y.copy()
        self.update_plot()
    
    def save_points(self, event):
        """保存剩余点到文件"""
        df = pd.DataFrame({'Depth': self.y, self.title: self.x})
        filename = f'filtered_{self.title}.csv'
        df.to_csv(filename, index=False)
        print(f"剩余 {len(self.x)} 个点已保存到 {filename}")
    
    def next_curve(self, event):
        """切换到下一条曲线"""
        self.should_continue = True
        plt.close(self.fig)
        
    def previous_curve(self, event):
        """返回上一条曲线"""
        self.should_continue = False
        plt.close(self.fig)
        
    def update_plot(self):
        """更新散点图"""
        self.scatter.set_offsets(np.column_stack([self.x, self.y]))
        self.ax.set_title(f'{self.title} - 剩余点数: {len(self.x)} (原始点数: {len(self.original_x)})')
        self.fig.canvas.draw_idle()

def main():
    try:
        # 尝试不同的编码方式读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(r'D:\AAAA工作-研一\3济阳凹陷\5超压预测\1-传统方法-Eaton\1-测井数据Resform\1-Reform导出测井数据\N组处理后数据\N873.csv', encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"使用 {encoding} 编码读取文件时出错: {str(e)}")
                continue
                
        if df is None:
            raise ValueError("无法读取文件，请检查文件编码")
        
        # 预处理数据
        processor = WellLogProcessor(df)
        processed_df = processor.filter_anomalies()
        
        # 获取所有数值型测井曲线列（除了Depth）
        log_columns = []
        for col in processed_df.columns:
            if col != 'Depth' and pd.api.types.is_numeric_dtype(processed_df[col]):
                log_columns.append(col)
        
        # 依次处理每条曲线
        current_index = 0
        while current_index < len(log_columns):
            column = log_columns[current_index]
            if column in processor.filtered_data:  # 只处理已经过滤过的曲线
                editor = InteractiveScatterEditor(
                    processor.filtered_data[column][column],
                    processor.filtered_data[column]['Depth'],
                    title=column
                )
                plt.show()
                
                # 根据should_continue决定是前进还是后退
                if not editor.should_continue and current_index > 0:
                    current_index -= 1
                else:
                    current_index += 1
        
        # 合并所有过滤后的数据
        final_df = processor.merge_filtered_data()
        if final_df is not None:
            final_df.to_csv('filtered_all_logs.csv', index=False)
            print(f"所有测井曲线数据已合并保存到 filtered_all_logs.csv，共 {len(final_df)} 个深度点")
            
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()