import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import os
from delet_sca import WellLogProcessor, InteractiveScatterEditor

class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style='Modern.TButton')

class WellLogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("测井数据处理系统 made by Jorlin")
        self.root.geometry("400x500")
    
        # 设置主题色
        self.primary_color = "#007AFF"  # iOS蓝色
        self.bg_color = "#F2F2F7"      # iOS浅灰背景色
        self.text_color = "#000000"    # 黑色文字
        
        # 初始化变量
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.a_value = tk.DoubleVar(value=-0.0001995)
        self.b_value = tk.DoubleVar(value=5.0099)
        
        # 配置样式
        self.setup_styles()
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        self.control_frame = ttk.Frame(self.main_frame, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # 文件选择部分
        self.create_file_section()
        
        # 参数设置部分
        self.create_parameter_section()
        
        # 创建右侧图形显示区域
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 初始化图形
        self.fig = None
        self.canvas = None
        
    def setup_styles(self):
        """设置自定义样式"""
        style = ttk.Style()
        
        # 配置主题
        style.theme_use('clam')
        
        # 配置按钮样式
        style.configure('Modern.TButton',
                       background=self.primary_color,
                       foreground='white',
                       padding=10,
                       font=('SF Pro Display', 10))
        
        # 配置标签样式
        style.configure('Modern.TLabel',
                       font=('SF Pro Display', 10),
                       padding=5)
        
        # 配置输入框样式
        style.configure('Modern.TEntry',
                       padding=5,
                       font=('SF Pro Display', 10))
        
    def create_file_section(self):
        """创建文件选择部分"""
        file_frame = ttk.LabelFrame(self.control_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 输入文件选择
        ttk.Label(file_frame, text="输入文件:").pack(anchor=tk.W)
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        # 使用Text控件替代Entry
        self.input_text = tk.Text(input_frame, height=3, width=30)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ModernButton(input_frame, text="浏览", command=self.select_input_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 输出目录选择
        ttk.Label(file_frame, text="输出目录:").pack(anchor=tk.W)
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        # 使用Text控件替代Entry
        self.output_text = tk.Text(output_frame, height=3, width=30)
        self.output_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ModernButton(output_frame, text="浏览", command=self.select_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
    def create_parameter_section(self):
        """创建参数设置部分"""
        param_frame = ttk.LabelFrame(self.control_frame, text="正常AC趋势线参数设置", padding="10")
        param_frame.pack(fill=tk.X)
        
        # a值设置
        ttk.Label(param_frame, text="a值:").pack(anchor=tk.W)
        ttk.Entry(param_frame, textvariable=self.a_value).pack(fill=tk.X, pady=5)
        
        # b值设置
        ttk.Label(param_frame, text="b值:").pack(anchor=tk.W)
        ttk.Entry(param_frame, textvariable=self.b_value).pack(fill=tk.X, pady=5)
        
        # 处理按钮
        ModernButton(param_frame, text="开始处理", command=self.process_data).pack(fill=tk.X, pady=10)
        
    def select_input_file(self):
        """选择输入文件"""
        filename = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            self.input_text.delete(1.0, tk.END)
            self.input_text.insert(tk.END, filename)
            
    def select_output_dir(self):
        """选择输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir.set(dirname)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, dirname)
            
    def process_data(self):
        """处理数据"""
        input_path = self.input_text.get(1.0, tk.END).strip()
        output_path = self.output_text.get(1.0, tk.END).strip()
        
        if not input_path:
            messagebox.showerror("错误", "请选择输入文件")
            return
            
        if not output_path:
            messagebox.showerror("错误", "请选择输出目录")
            return
            
        try:
            # 读取数据
            df = pd.read_csv(input_path)
            
            # 预处理数据
            processor = WellLogProcessor(df)
            processed_df = processor.filter_anomalies()
            
            # 获取所有测井曲线列（除了Depth）
            log_columns = [col for col in processed_df.columns if col != 'Depth']
            
            # 依次处理每条曲线
            for column in log_columns:
                if column in processor.filtered_data:
                    editor = InteractiveScatterEditor(
                        processor.filtered_data[column][column],
                        processor.filtered_data[column]['Depth'],
                        title=column,
                        a=self.a_value.get(),
                        b=self.b_value.get()
                    )
                    plt.show()
            
            # 合并所有过滤后的数据
            final_df = processor.merge_filtered_data()
            if final_df is not None:
                # 获取输入文件名并添加filtered后缀
                input_filename = os.path.basename(input_path)
                base_name = os.path.splitext(input_filename)[0]
                output_filename = f"{base_name}_filtered.csv"
                output_file = os.path.join(output_path, output_filename)
                
                final_df.to_csv(output_file, index=False)
                messagebox.showinfo("成功", f"处理完成！\n结果已保存到: {output_file}")
                
        except Exception as e:
            messagebox.showerror("错误", f"处理过程中出现错误：{str(e)}")

def main():
    root = tk.Tk()
    app = WellLogGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 