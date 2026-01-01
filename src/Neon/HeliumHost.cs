using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;
        private const int WM_MOUSEWHEEL = 0x020A;  // Vertical
        private const int WM_MOUSEHWHEEL = 0x020E; // Horizontal

        public HeliumHost()
        {
            this.Focusable = true;
        }

        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);

            this.Focus();
        }

        protected override HandleRef BuildWindowCore(HandleRef hwndParent)
        {
            _hwnd = NativeMethods.CreateWindowEx(
            0,
            "STATIC",
            "",
            NativeMethods.WS_CHILD |
            NativeMethods.WS_VISIBLE |
            NativeMethods.WS_CLIPCHILDREN |
            NativeMethods.WS_CLIPSIBLINGS,
            0,
            0,
            (int)ActualWidth,
            (int)ActualHeight,
            hwndParent.Handle,
            IntPtr.Zero,
            IntPtr.Zero,
            IntPtr.Zero);

            if (_hwnd == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Helium host window.");

            HeliumNative.He_Initialize(_hwnd, 0);

            return new HandleRef(this, _hwnd);
        }

        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            HeliumNative.He_Shutdown();
            NativeMethods.DestroyWindow(hwnd.Handle);
            _hwnd = IntPtr.Zero;
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(sizeInfo);

            if (_hwnd != IntPtr.Zero)
            {
                // Get DPI scale factor
                double dpiScaleX = 1.0;
                double dpiScaleY = 1.0;

                PresentationSource source = PresentationSource.FromVisual(this);
                if (source != null && source.CompositionTarget != null)
                {
                    dpiScaleX = source.CompositionTarget.TransformToDevice.M11;
                    dpiScaleY = source.CompositionTarget.TransformToDevice.M22;
                }

                // Convert logical size to physical pixels
                int width = (int)(sizeInfo.NewSize.Width * dpiScaleX);
                int height = (int)(sizeInfo.NewSize.Height * dpiScaleY);

                HeliumNative.He_Resize(width, height);
            }
        }

        protected override IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
        {
            switch (msg)
            {
                case WM_MOUSEWHEEL:
                    {
                        // Vertical Wheel
                        // Extract high order word for delta
                        int rawDelta = (short)((wParam.ToInt64() >> 16) & 0xFFFF);
                        float delta = rawDelta / 120.0f;

                        // Pass to Engine (x=0, y=delta)
                        HeliumNative.He_ProcessMouseWheel(0.0f, delta);

                        handled = true;
                        return IntPtr.Zero;
                    }

                case WM_MOUSEHWHEEL:
                    {
                        // Horizontal Wheel
                        int rawDelta = (short)((wParam.ToInt64() >> 16) & 0xFFFF);
                        float delta = rawDelta / 120.0f;

                        // Pass to Engine (x=delta, y=0)
                        // Note: Depending on your camera/scroll logic, you might want to invert this (-delta)
                        HeliumNative.He_ProcessMouseWheel(-delta, 0.0f);

                        handled = true;
                        return IntPtr.Zero;
                    }
            }

            return base.WndProc(hwnd, msg, wParam, lParam, ref handled);
        }

        // 세로 휠(Vertical)은 기존처럼 OnMouseWheel에서 처리하거나
        // 통일성을 위해 여기서 WM_MOUSEWHEEL (0x020A)을 함께 처리해도 됩니다.
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            // 세로 휠 처리
            base.OnMouseWheel(e);

            if (!e.Handled)
            {
                float delta = e.Delta / 120.0f;
                HeliumNative.He_ProcessMouseWheel(0.0f, delta);
                e.Handled = true;
            }
        }
    }
}
