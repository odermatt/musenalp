package org.esa.snap.musenalp;

import org.esa.snap.core.gpf.ui.DefaultSingleTargetProductDialog;
import org.esa.snap.core.gpf.ui.OperatorMenu;
import org.esa.snap.ui.AppContext;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class CustomSingleTargetProductDialog extends DefaultSingleTargetProductDialog {
    CustomSingleTargetProductDialog(String operatorName, AppContext appContext, String title, String helpID) {
        super(operatorName, appContext, title, helpID, true);
    }

    public int reshow() {
        try {
            Method m = getClass().getSuperclass().getDeclaredMethod("initForm");
            m.setAccessible(true);
            m.invoke(this, (Object[]) null);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }
        return super.show();
    }

}
