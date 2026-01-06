using System;
using System.Runtime.InteropServices;
using System.Windows.Interop;
using System.Windows;
using System.Windows.Input;
using System.Threading;     // [추가] 스레드 사용
using System.Diagnostics;   // [추가] Stopwatch 사용

namespace Neon.Controls
{
    public sealed class HeliumHost : HwndHost
    {
        private IntPtr _hwnd = IntPtr.Zero;

        // 렌더링 스레드 관련 변수
        private Thread? _renderThread;
        private volatile bool _isRunning = false;

        // 리사이징 동기화를 위한 변수 (volatile: 스레드 간 최신값 보장)
        private volatile bool _isResizePending = false;
        private volatile int _newWidth = 0;
        private volatile int _newHeight = 0;

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

            // [변경 1] 여기서 He_Initialize를 호출하지 않습니다.
            // UI 스레드에서 초기화하면 렌더 스레드에서 그릴 수 없기 때문입니다.

            // [변경 2] 렌더링 스레드 시작
            _isRunning = true;
            _renderThread = new Thread(RenderLoop);
            _renderThread.Name = "Helium Render Thread";
            _renderThread.IsBackground = true; // 앱 종료 시 자동 종료
            _renderThread.Start();

            return new HandleRef(this, _hwnd);
        }

        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            // [변경 3] 스레드 안전 종료
            _isRunning = false;

            if (_renderThread != null && _renderThread.IsAlive)
            {
                _renderThread.Join(1000); // 최대 1초 대기
            }

            // DestroyWindow는 UI 스레드에서 해야 함
            NativeMethods.DestroyWindow(hwnd.Handle);
            _hwnd = IntPtr.Zero;
        }

        // [핵심] 별도 스레드에서 돌아가는 렌더링 루프
        private void RenderLoop()
        {
            // 1. 이 스레드에서 초기화 (OpenGL Context 생성)
            // HWND는 이미 BuildWindowCore에서 생성되었으므로 안전하게 접근 가능
            HeliumNative.He_Initialize(_hwnd, 0);

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            long lastTicks = 0;

            while (_isRunning)
            {
                // 2. 리사이즈 처리 (Pending 상태일 때만 수행)
                if (_isResizePending)
                {
                    HeliumNative.He_Resize(_newWidth, _newHeight);
                    _isResizePending = false;
                }

                // 3. 델타 타임 계산
                long currentTicks = stopwatch.ElapsedTicks;
                float dt = (float)(currentTicks - lastTicks) / Stopwatch.Frequency;
                lastTicks = currentTicks;

                // 4. 엔진 업데이트 및 렌더링
                try
                {
                    HeliumNative.He_Update(dt);
                    HeliumNative.He_Render();
                }
                catch (Exception ex)
                {
                    // 로그 남기기 (Console or Logger)
                    Debug.WriteLine($"Render Error: {ex.Message}");
                }

                // (선택 사항) CPU 점유율을 낮추려면 1ms 대기
                // Thread.Sleep(1); 
            }

            // 5. 루프 종료 시 정리 (Context 해제)
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

                    // [변경 4] 직접 He_Resize를 호출하지 않고, 값을 저장하고 플래그를 세웁니다.
                    // UI 스레드에서 OpenGL 함수를 호출하면 안 되기 때문입니다.
                    _newWidth = (int)(sizeInfo.NewSize.Width * scaleX);
                    _newHeight = (int)(sizeInfo.NewSize.Height * scaleY);
                    _isResizePending = true;
                }
            }
        }
    }
}