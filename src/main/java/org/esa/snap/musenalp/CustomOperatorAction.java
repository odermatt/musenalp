package org.esa.snap.musenalp;

import com.bc.ceres.binding.Property;
import com.bc.ceres.binding.PropertySet;
import com.bc.ceres.swing.binding.BindingContext;
import org.esa.snap.core.gpf.ui.DefaultOperatorAction;

import java.awt.event.ActionEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class CustomOperatorAction extends DefaultOperatorAction {

    private static final Set<String> KNOWN_KEYS = new HashSet<>(Arrays.asList("displayName", "operatorName", "dialogTitle", "helpId", "targetProductNameSuffix"));
    private CustomSingleTargetProductDialog dialog;
    private Property[] originalProperties;


    public static CustomOperatorAction create(Map<String, Object> properties) {
        CustomOperatorAction action = new CustomOperatorAction();
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            if (KNOWN_KEYS.contains(entry.getKey())) {
                action.putValue(entry.getKey(), entry.getValue());
            }
        }
        return action;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (dialog == null) {
            dialog = new CustomSingleTargetProductDialog(getOperatorName(), getAppContext(),
                                                         getDialogTitle(), getHelpId());
            if (getTargetProductNameSuffix() != null) {
                dialog.setTargetProductNameSuffix(getTargetProductNameSuffix());
            }
        }
        final BindingContext bindingContext = dialog.getBindingContext();
        final PropertySet propertySet = bindingContext.getPropertySet();
        configurePropertySet(propertySet);

        bindingContext.addPropertyChangeListener("algorithm", new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                String algorithm = (String) evt.getNewValue();
                if (!algorithm.isEmpty()) {
                    for (Property property : originalProperties) {
                        //TODO Reihenfolge der Properties beachten!
                        if (property.getName().startsWith(algorithm)) {
                            propertySet.addProperty(property);
                        }
                    }
                }
                dialog.getJDialog().pack();
                dialog.reshow();
            }
        });
        bindingContext.addPropertyChangeListener("cut", new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                boolean subsetValue = (boolean) evt.getNewValue();
                if (subsetValue) {
                    for (Property property : originalProperties) {
                        //TODO Reihenfolge der Properties beachten!
                        if (property.getName().startsWith("cut")) {
                            propertySet.addProperty(property);
                        }
                    }
                }
                dialog.getJDialog().pack();
                dialog.reshow();
            }
        });
        dialog.getJDialog().pack();
        dialog.show();
    }

    private void configurePropertySet(PropertySet propertySet) {
        originalProperties = propertySet.getProperties();
        for (Property property : originalProperties) {
            if (property.getName().contains("_")) {
                propertySet.removeProperty(property);
            }
        }
    }
}