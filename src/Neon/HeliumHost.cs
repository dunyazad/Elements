using System;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using System.Windows;
using System.Windows.Input;

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;

        public HeliumHost()
        {
            // 포커스 설정 불필요
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
                var source = PresentationSource.FromVisual(this);
                if (source?.CompositionTarget != null)
                {
                    double scaleX = source.CompositionTarget.TransformToDevice.M11;
                    double scaleY = source.CompositionTarget.TransformToDevice.M22;

                    HeliumNative.He_Resize(
                        (int)(sizeInfo.NewSize.Width * scaleX),
                        (int)(sizeInfo.NewSize.Height * scaleY));
                }
            }
        }
    }
}
