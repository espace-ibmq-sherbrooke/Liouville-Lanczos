

try:
    import quimb
except:
    print(
        """
        Failed to import quimb. This module is
        necessary for the execution of Tensor network methods.
        Install it from https://github.com/jcmgray/quimb .
        """)