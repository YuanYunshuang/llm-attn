
train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10),
        dict(type="SegPlotWriterHook", max_n_img=2, log_every=20)
    ]

lgln_train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10),
        dict(type="LGLNSegPlotWriterHook", max_n_img=2, log_every=100)
    ]

lgln_train_hooks_pseudo = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type='LGLNPseudoLabelHook'),
        dict(type="CheckPointsHook", epoch_every=10),
        # dict(type="LGLNSegPlotWriterHook", max_n_img=2, log_every=100)
    ]

test_hooks = [
    dict(type='HolisticSegResultHook'),
]

cvit_train_hooks = [
    dict(type='MemoryUsageHook'),
    dict(type='TrainTimerHook'),
    dict(type="CheckPointsHook", epoch_every=10),
]

cvit_test_hooks = [
    dict(type='CCVitResultHook'),
]
