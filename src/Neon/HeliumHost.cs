using System;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using System.Windows;
using System.Windows.Input;
using System.Threading;
using System.Diagnostics;

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;

        private Thread? _renderThread;
        private volatile bool _isRunning = false;

        // 리사이징 동기화를 위한 변수 (volatile: 스레드 간 최신값 보장)
        private volatile bool _isResizePending = false;
        private volatile int _newWidth = 0;
        private volatile int _newHeight = 0;

        public HeliumHost()
        {
        }

        protected override HandleRef BuildWindowCore(HandleRef hwndParent)
        {
            _hwnd = NativeMethods.CreateWindowEx(
                0,
                "STATIC",
                "",
                NativeMethods.WS_CHILD | NativeMethods.WS_VISIBLE | NativeMethods.WS_CLIPCHILDREN,
                0, 0,
                (int)ActualWidth, (int)ActualHeight,
                hwndParent.Handle,
                IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);

            if (_hwnd == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Helium host window.");

            _isRunning = true;
            _renderThread = new Thread(RenderLoop);
            _renderThread.Name = "Helium Render Thread";
            _renderThread.IsBackground = true;
            _renderThread.Start();

            return new HandleRef(this, _hwnd);
        }

        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            _isRunning = false;

            if (_renderThread != null && _renderThread.IsAlive)
            {
                _renderThread.Join(1000);
            }

            NativeMethods.DestroyWindow(hwnd.Handle);
            _hwnd = IntPtr.Zero;
        }

        private void RenderLoop()
        {
            HeliumNative.He_Initialize(_hwnd, 0);

            HeliumNative.He_InitializeScene3D();

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            long lastTicks = 0;

            while (_isRunning)
            {
                if (_isResizePending)
                {
                    HeliumNative.He_Resize(_newWidth, _newHeight);
                    _isResizePending = false;
                }

                long currentTicks = stopwatch.ElapsedTicks;
                float dt = (float)(currentTicks - lastTicks) / Stopwatch.Frequency;
                lastTicks = currentTicks;

                try
                {
                    HeliumNative.He_Update(dt);
                    HeliumNative.He_Render();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Render Error: {ex.Message}");
                }

                Thread.Sleep(1); 
            }

            HeliumNative.He_Shutdown();
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(sizeInfo);

            if (_hwnd != IntPtr.Zero)
            {
                var source = PresentationSource.FromVisual(this);
                if (source?.CompositionTarget != null)
                {
                    double scaleX = source.CompositionTarget.TransformToDevice.M11;
                    double scaleY = source.CompositionTarget.TransformToDevice.M22;

                    _newWidth = (int)(sizeInfo.NewSize.Width * scaleX);
                    _newHeight = (int)(sizeInfo.NewSize.Height * scaleY);
                    _isResizePending = true;
                }
            }
        }
    }
}